import pandas as pd 
import torch
import numpy as np
import pandas as pd
from transformers import (
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    ByT5Tokenizer,
    PreTrainedTokenizer,
    T5TokenizerFast as T5Tokenizer,
    MT5TokenizerFast as MT5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer
)
import csvs
import csv
import jsonlines
import torchmetrics

from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelWithLMHead, AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import random 
import tqdm

torch.cuda.empty_cache()
pl.seed_everything(42)


class PyTorchDataModule(Dataset):
    """  PyTorch Dataset class  """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        validation: bool = False

    ):
        """
        initiates a PyTorch Dataset Module for input data
        Args:
            data (pd.DataFrame): input pandas dataframe. Dataframe must have 2 column --> "source_text" and "target_text"
            tokenizer (PreTrainedTokenizer): a PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.validation = validation

    def __len__(self):
        """ returns length of data """
        return len(self.data)

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5/MT5 model"""

        data_row = self.data.iloc[index]
        source_text = data_row["source_text"]

        source_text_encoding = self.tokenizer(
            source_text,
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        target_text_encoding = self.tokenizer(
            data_row["target_text"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = target_text_encoding["input_ids"]
        labels[
            labels == 0
        ] = -100  # to make sure we have correct labels for T5 text generation

        if self.validation:
            return dict(
                source_text_input_ids=source_text_encoding["input_ids"].flatten(),
                source_text_attention_mask=source_text_encoding["attention_mask"].flatten(),
                labels=labels.flatten(),
                labels_attention_mask=target_text_encoding["attention_mask"].flatten(),
                source_text=source_text,
                target_text=data_row["target_text"]
            )
        
        else:
            return dict(
                source_text_input_ids=source_text_encoding["input_ids"].flatten(),
                source_text_attention_mask=source_text_encoding["attention_mask"].flatten(),
                labels=labels.flatten(),
                labels_attention_mask=target_text_encoding["attention_mask"].flatten(),
            )


class LightningDataModule(pl.LightningDataModule):
    """ PyTorch Lightning data class """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        num_workers: int = 2,
    ):
        """
        initiates a PyTorch Lightning Data Module
        Args:
            train_df (pd.DataFrame): training dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            test_df (pd.DataFrame): validation dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            tokenizer (PreTrainedTokenizer): PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            batch_size (int, optional): batch size. Defaults to 4.
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = PyTorchDataModule(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )
        self.test_dataset = PyTorchDataModule(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
            validation=True
        )

    def train_dataloader(self):
        """ training dataloader """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """ test dataloader """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """ validation dataloader """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class LightningModel(pl.LightningModule):
    """ PyTorch Lightning Model class"""

    def __init__(
        self,
        tokenizer,
        model,
        outputdir: str = "outputs",
        save_only_last_epoch: bool = False,
    ):
        """
        initiates a PyTorch Lightning Model
        Args:
            tokenizer : T5/MT5/ByT5 tokenizer
            model : T5/MT5/ByT5 model
            outputdir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
            save_only_last_epoch (bool, optional): If True, save just the last epoch else models are saved for every epoch
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.outputdir = outputdir
        self.average_training_loss = None
        self.average_validation_loss = None
        self.save_only_last_epoch = save_only_last_epoch
        self.valid_acc = torchmetrics.Accuracy()


    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        """ training step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )
        return loss

    def validation_step(self, batch, batch_size):
        """ validation step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        
        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log(
            "val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )

        gold = batch['target_text']
        guess = self.tokenizer.batch_decode(
            self.model.generate(
                input_ids=batch["source_text_input_ids"],
                attention_mask=batch["source_text_attention_mask"],
                min_length=0,
                num_beams=5,
                num_return_sequences=1,
            ),
            skip_special_tokens=True,
        )


        acc = torch.tensor(
            [
                str(a) in gold for a in guess
            ]
        ).long()


        self.valid_acc(acc, torch.ones_like(acc))
        self.log(
            "valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True
        )


        return loss

    def test_step(self, batch, batch_size):
        """ test step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """ configure optimizers """
        return AdamW(self.parameters(), lr=0.0001)

    def training_epoch_end(self, training_step_outputs):
        """ save tokenizer and model on epoch end """
        self.average_training_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            4,
        )
        path = f"{self.outputdir}/simplet5-epoch-{self.current_epoch}-train-loss-{str(self.average_training_loss)}-val-loss-{str(self.average_validation_loss)}"
        if self.save_only_last_epoch:
            if self.current_epoch == self.trainer.max_epochs - 1:
                self.tokenizer.save_pretrained(path)
                self.model.save_pretrained(path)
        else:
            self.tokenizer.save_pretrained(path)
            self.model.save_pretrained(path)

    def validation_epoch_end(self, validation_step_outputs):
        _loss = [x.cpu() for x in validation_step_outputs]
        self.average_validation_loss = np.round(
            torch.mean(torch.stack(_loss)).item(),
            4,
        )


class SimpleT5:
    """ Custom SimpleT5 class """

    def __init__(self) -> None:
        """ initiates SimpleT5 class """
        pass

    def from_pretrained(self, model_type="t5", model_name="t5-base") -> None:
        """
        loads T5/MT5 Model model for training/finetuning
        Args:
            model_type (str, optional): "t5" or "mt5" . Defaults to "t5".
            model_name (str, optional): exact model architecture name, "t5-base" or "t5-large". Defaults to "t5-base".
        """
        if model_type == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(f"{model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        elif model_type == "mt5":
            self.tokenizer = MT5Tokenizer.from_pretrained(f"{model_name}")
            self.model = MT5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        elif model_type == "byt5":
            self.tokenizer = ByT5Tokenizer.from_pretrained(f"{model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )

        elif model_type == "facebook/bart-base":
            self.tokenizer = BartTokenizer.from_pretrained(f"{model_name}")
            self.model = BartForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        batch_size: int = 8,
        max_epochs: int = 5,
        use_gpu: int = 1,
        outputdir: str = "outputs",
        early_stopping_patience_epochs: int = 0,  # 0 to disable early stopping feature
        precision=32,
        logger="default",
        dataloader_num_workers: int = 2,
        save_only_last_epoch: bool = False,
    ):
        """
        trains T5/MT5 model on custom dataset
        Args:
            train_df (pd.DataFrame): training datarame. Dataframe must have 2 column --> "source_text" and "target_text"
            eval_df ([type], optional): validation datarame. Dataframe must have 2 column --> "source_text" and "target_text"
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
            batch_size (int, optional): batch size. Defaults to 8.
            max_epochs (int, optional): max number of epochs. Defaults to 5.
            use_gpu (bool, optional): if True, model uses gpu for training. Defaults to True.
            outputdir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
            early_stopping_patience_epochs (int, optional): monitors val_loss on epoch end and stops training, if val_loss does not improve after the specied number of epochs. set 0 to disable early stopping. Defaults to 0 (disabled)
            precision (int, optional): sets precision training - Double precision (64), full precision (32) or half precision (16). Defaults to 32.
            logger (pytorch_lightning.loggers) : any logger supported by PyTorch Lightning. Defaults to "default". If "default", pytorch lightning default logger is used.
            dataloader_num_workers (int, optional): number of workers in train/test/val dataloader
            save_only_last_epoch (bool, optional): If True, saves only the last epoch else models are saved at every epoch
        """
        self.data_module = LightningDataModule(
            train_df,
            eval_df,
            self.tokenizer,
            batch_size=batch_size,
            source_max_token_len=source_max_token_len,
            target_max_token_len=target_max_token_len,
            num_workers=dataloader_num_workers,
        )

        self.T5Model = LightningModel(
            tokenizer=self.tokenizer,
            model=self.model,
            outputdir=outputdir,
            save_only_last_epoch=save_only_last_epoch,
        )

        # add callbacks
        callbacks = [TQDMProgressBar(refresh_rate=5)]

        if early_stopping_patience_epochs > 0:
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=early_stopping_patience_epochs,
                verbose=True,
                mode="min",
            )
            callbacks.append(early_stop_callback)

        # add gpu support
        gpus = use_gpu
        print(gpus)

        # add logger
        loggers = True if logger == "default" else logger

        # prepare trainer
        trainer = pl.Trainer(
            logger=loggers,
            callbacks=callbacks,
            max_epochs=max_epochs,
            gpus=gpus,
            precision=precision,
            log_every_n_steps=1,
            strategy='ddp'
        )

        # fit trainer
        trainer.fit(self.T5Model, self.data_module)

    def load_model(
        self, model_type: str = "t5", model_dir: str = "outputs", use_gpu: bool = False
    ):
        """
        loads a checkpoint for inferencing/prediction
        Args:
            model_type (str, optional): "t5" or "mt5". Defaults to "t5".
            model_dir (str, optional): path to model directory. Defaults to "outputs".
            use_gpu (bool, optional): if True, model uses gpu for inferencing/prediction. Defaults to True.
        """
        if model_type == "t5":
            self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = T5Tokenizer.from_pretrained(f"{model_dir}")
        elif model_type == "mt5":
            self.model = MT5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = MT5Tokenizer.from_pretrained(f"{model_dir}")
        elif model_type == "byt5":
            self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = ByT5Tokenizer.from_pretrained(f"{model_dir}")
        elif model_type == "facebook/bart-base":
            self.model = BartForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = BartTokenizer.from_pretrained(f"{model_dir}")

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:1")
            else:
                raise "exception ---> no gpu found. set use_gpu=False, to use CPU"
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

    def predict(
        self,
        source_text: str,
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ):
        """
        generates prediction for T5/MT5 model
        Args:
            source_text (str): any text for generating predictions
            max_length (int, optional): max token length of prediction. Defaults to 512.
            num_return_sequences (int, optional): number of predictions to be returned. Defaults to 1.
            num_beams (int, optional): number of beams. Defaults to 2.
            top_k (int, optional): Defaults to 50.
            top_p (float, optional): Defaults to 0.95.
            do_sample (bool, optional): Defaults to True.
            repetition_penalty (float, optional): Defaults to 2.5.
            length_penalty (float, optional): Defaults to 1.0.
            early_stopping (bool, optional): Defaults to True.
            skip_special_tokens (bool, optional): Defaults to True.
            clean_up_tokenization_spaces (bool, optional): Defaults to True.
        Returns:
            list[str]: returns predictions
        """
        input_ids = self.tokenizer.encode(
            source_text, return_tensors="pt", add_special_tokens=True
        )
        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
        )
        preds = [
            self.tokenizer.decode(
                g,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for g in generated_ids
        ]
        return preds


# ~~~~~~~trainin_dataset~~~~~~~~~~~~~~~
# train_spider = pd.read_csv('train_normalized.csv')
# dev_spider = pd.read_csv('dev_normalized.csv')

# train_data = [train_spider['question_id'], train_spider['question_text'], train_spider['program']]
# train_data_header =  ["id","source_text", "target_text"]

# dev_data = [dev_spider['question_id'], dev_spider['question_text'], dev_spider['program']]
# dev_data_header =  ["id","source_text", "target_text"]

# train_data_df = pd.concat(train_data, axis=1, keys=train_data_header)
# dev_data_df = pd.concat(dev_data, axis=1, keys=dev_data_header)
# train_data_df.to_csv('train_data_df.csv', index=False)
# dev_data_df.to_csv('dev_data_df.csv', index=False)
# ~~~~~~~model_training~~~~~~~~~~~~~~~

# model = SimpleT5()
# model.from_pretrained("t5","t5-base")

# model.train(train_df=train_data_df, # pandas dataframe with 2 columns: source_text & target_text
#             eval_df=dev_data_df, # pandas dataframe with 2 columns: source_text & target_text
#             source_max_token_len = 128, 
#             target_max_token_len = 256,
#             batch_size = 32,
#             max_epochs = 8,
#             use_gpu = 4,
#             outputdir = "outputs",
#             early_stopping_patience_epochs = 0,
#             precision = 32,
#             dataloader_num_workers = 16,
#             )


# ~~~~~~~model_evaluation~~~~~~~~~~~~~~~

# model = SimpleT5()

# model.load_model("t5","/root/sparqling-queries/data/break/logical-forms-fixed/outputs/simplet5-epoch-7-train-loss-0.1483-val-loss-0.1631", use_gpu=True)
# model_output = model.predict("show me the afternoon flights from washington to boston ", num_return_sequences=5, num_beams=5)
# print(model_output[0])

# ~~~~~~~~~~~predict~~~~~~~~~~~~~~~~~~~~~
# evl_df = pd.read_csv('dev_normalized.csv')
# evl_data = [evl_df['question_id'], evl_df['question_text'], evl_df['program']]
# evl_data_header =  ["id","source_text", "target_text"]
# evl_data_df = pd.concat(evl_data, axis=1, keys=evl_data_header)

# sample = evl_data_df[:100]
# sample_source = sample['source_text'].tolist()
# sample_target = sample['target_text'].tolist()
# sample_id = sample['id'].tolist()

# em = 0
# wrong = 0
# formlized_em = 0
# model_outputs = []
# alternatives = []

# pbar = tqdm.tqdm(zip(sample_source, sample_target),total=len(sample_source))
# for source, target in pbar:
#     model_output = model.predict(source, num_return_sequences=5, num_beams=5)
#     # print(model_output)
#     alternative = random.choice(model_output[1:])

#     alternatives.append(alternative)
#     model_outputs.append(model_output[0])

#     if str(model_output[0]) == target:
#         em += 1
#     else:
#         wrong += 1
    
#     pbar.set_postfix(em=em, wrong=wrong, accuracy=(em/(em+wrong)))


# print('~~~~~~~em~~~~~~~~~~~', em/(em+wrong))
# print('~~~~accuracy~~~~~~~', em/4000)
   

# ~~~~~~~alternative_dataset~~~~~~~~~~~~~~

device = 0
print('~~~~~~train_device~~~~~~', device)
train_spider = pd.read_csv('train_normalized.csv')

train_data = [train_spider['question_id'], train_spider['question_text'], train_spider['program']]
train_data_header =  ["id","source_text", "target_text"]
train_data_df = pd.concat(train_data, axis=1, keys=train_data_header)

if device == 0:
    sample = train_data_df[:5000]
else:
    sample = train_data_df[device*5000:(device+1)*5000]

sample_id = sample['id'].tolist()
sample_source = sample['source_text'].tolist()
sample_target = sample['target_text'].tolist()

model_outputs = []

model = SimpleT5()

model.load_model("t5","/root/sparqling-queries/data/break/logical-forms-fixed/outputs/simplet5-epoch-7-train-loss-0.1483-val-loss-0.1631", use_gpu=True)

alternatives = []

for source in tqdm.tqdm(sample_source, total=len(sample_source)):
    outputs = model.predict(source, num_return_sequences=5, num_beams=5)
    # print(model_outputs)
    output = outputs[0]
    alt = random.choice(outputs[1:])
    assert output, "empty model output"
    model_outputs.append(output)
    alternatives.append(alt)

alternative_data = list(zip(sample_id, sample_source, model_outputs, alternatives))
alternative_data_header =  ["id","source_text", "output_text", "alt_text"]
alternative_data_df = pd.DataFrame(alternative_data, columns=alternative_data_header)
alternative_data_df.to_csv('dev_alter_dataset_V3_device_%s.csv'%(device),index=False)


# df1 = pd.read_csv('train_alter_dataset_V3_device_0.csv')
# df2 = pd.read_csv('train_alter_dataset_V3_device_1.csv')
# df3 = pd.read_csv('train_alter_dataset_V3_device_2.csv')
# df4 = pd.read_csv('train_alter_dataset_V3_device_3.csv')
# df5 = pd.read_csv('train_alter_dataset_V2_device_5.csv')
# df5 = pd.read_csv('alter_dataset_device_5.csv')
# df6 = pd.read_csv('alter_dataset_device_6.csv')
# df7 = pd.read_csv('alter_dataset_device_7.csv')
# df8 = pd.read_csv('alter_dataset_device_8.csv')
# df9 = pd.read_csv('alter_dataset_device_9.csv')

# train_df = pd.concat([df1,df2,df3,df4])

# train_df.to_csv('train_alter_v3.csv', index=False)

# dev_df1 = pd.read_csv('dev_alter_dataset_V3_device_0.csv')
# dev_df2 = pd.read_csv('dev_alter_dataset_V3_device_1.csv')

# dev_df = pd.concat([dev_df1,dev_df2])

# dev_df.to_csv('dev_alter_v3.csv', index=False)

    
# # Open a csv reader called DictReader

# def make_json(csvFilePath, jsonFilePath):
     
#     # create a dictionary
#     data = []
     
#     # Open a csv reader called DictReader
#     with open(csvFilePath, encoding='utf-8') as csvf:
#         csvReader = csv.DictReader(csvf)
         
#         # Convert each row into a dictionary
#         # and add it to data
#         for rows in csvReader:
#             line = {}
#             line['id'] = rows['id']
#             line['input'] = rows['source_text']
#             line['prediction'] = rows['output_text']
#             line['alternatives'] = rows['alt_text']

#             data.append(line)

 
#     # Open a json writer, and use the json.dumps()
#     # function to dump data
#     with jsonlines.open(jsonFilePath, "w") as f:
#         f.write_all(data)

# csvFilePath = 'train_alter_v3.csv'
# jsonFilePath = 'train_alter_v3.jsonl'
 
# # # # Call the make_json function
# make_json(csvFilePath, jsonFilePath)


# csvFilePath = 'dev_alter_v3.csv'
# jsonFilePath = 'dev_alter_v3.jsonl'
 
# # # # Call the make_json function
# make_json(csvFilePath, jsonFilePath)

#~
# ~~~~~~~~~~~~train data clean ~~~~~~~~~~~~~~


# train_data = [train_spider['question_id'], train_spider['question_text'], train_spider['program']]
# train_data_header =  ["id","source_text", "target_text"]
# train_data_df = pd.concat(train_data, axis=1, keys=train_data_header)

#～～～～～～～～～～evaluate the alter dataset～～～～～～～～～～～～～～～～～～～～～～～
# model = SimpleT5()

# model.load_model("t5","/root/sparqling-queries/data/break/logical-forms-fixed/outputs/simplet5-epoch-7-train-loss-0.1483-val-loss-0.1631", use_gpu=True)
# evl_df = pd.read_csv('train_alter_v3.csv')
# evl_data = [evl_df['source_text'], evl_df['output_text']]
# evl_data_header =  ["source_text", "target_text"]
# evl_data_df = pd.concat(evl_data, axis=1, keys=evl_data_header)

# sample = evl_data_df[:100]
# sample_source = sample['source_text'].tolist()
# sample_target = sample['target_text'].tolist()

# em = 0
# wrong = 0

# pbar = tqdm.tqdm(zip(sample_source, sample_target),total=len(sample_source))
# for source, target in pbar:
#     model_output = model.predict(source, num_return_sequences=1, num_beams=5)

#     if str(model_output[0]) == target:
#         em += 1
#     else:
#         wrong += 1
    
#     pbar.set_postfix(em=em, wrong=wrong, accuracy=(em/(em+wrong)))

