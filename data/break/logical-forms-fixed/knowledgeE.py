import math
from argparse import ArgumentParser
from copy import deepcopy

import torch
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

import pytorch_lightning as pl
import torch
from higher.patch import monkeypatch as make_functional
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_constant_schedule,
    get_linear_schedule_with_warmup,
)
import random

import jsonlines
import numpy as np
from torch.utils.data import Dataset


def batch_it(seq, num=1):
    assert num > 0
    batch = []
    for e in seq:
        if len(batch) == num:
            yield batch
            batch = [e]
        else:
            batch.append(e)
    yield batch


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class BartSeq2SeqAugmented(LightningModule):
    @staticmethod
    #  下面是所有的hyperparameter
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--train_data_path",
            type=str,
            default="/root/sparqling-queries/data/break/logical-forms-fixed/train_alter.jsonl",
        )
        parser.add_argument(
            "--dev_data_path",
            type=str,
            default="/root/sparqling-queries/data/break/logical-forms-fixed/dev_alter.jsonl",
        )
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--lr_alpha", type=float, default=1e-1)
        parser.add_argument("--max_length", type=int, default=32)
        parser.add_argument("--weight_decay", type=int, default=0.01)
        parser.add_argument("--total_num_updates", type=int, default=200000)
        parser.add_argument("--warmup_updates", type=int, default=1000)
        parser.add_argument("--num_workers", type=int, default=0)

        parser.add_argument("--model_name", type=str, default="t5-base")
        parser.add_argument("--eps", type=float, default=0.1)
        parser.add_argument(
            "--model_checkpoint",
            type=str,
            default="/root/sparqling-queries/data/break/logical-forms-fixed/lightning_logs/version_3/checkpoints/epoch=9-step=6899.ckpt",
        )

        parser.add_argument("--margin_kl_max", type=float, default=1e-3)
        parser.add_argument("--margin_kl_min", type=float, default=1e-5)
        parser.add_argument("--margin_lp_max", type=float, default=1e-3)
        parser.add_argument("--margin_lp_min", type=float, default=1e-7)
        parser.add_argument("--max_scale", type=float, default=1)
        parser.add_argument("--p", type=float, default=2)
        parser.add_argument(
            "--divergences", type=str, choices=["kl", "lp", "both"], default="kl"
        )
        parser.add_argument("--use_views", action="store_true")

        return parser

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # tokenizer可以替换成我们的
        # self.tokenizer = AutoTokenizer.from_pretrained('Salesforce/grappa_large_jnt')

        # model 参考 text2qdmr 里 infer的写法 
        # self.model = registry.construct('model', self.config['model'], preproc=self.model_preproc, device=self.device)
        # self.model.to(self.device)
        # self.model.eval()
        # saver = saver_mod.Saver({"model": model})
        # last_step = saver.restore(logdir, step=step, map_location=self.device, item_keys=["model"])

        print(self.hparams.model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_name)
        # self model 也可以替换
        self.model = BartSeq2Seq.load_from_checkpoint(
            self.hparams.model_checkpoint
        ).model.eval()
        # model.eval()可以
        self.learner = OneShotLearner(
            self.model,
            #dimension可以写上去 我们的是50265 或者，我们的是
            # self.model.encoder.embeddings.word_embeddings.weight.data.shape[0]

            vocab_dim=self.model.model.shared.weight.data.shape[0],
            # embedding_dim 可能是1024
            embedding_dim=self.model.model.shared.weight.data.shape[1],
            hidden_dim=128,
            condition_dim=1024,
            #include_set可能是控制有哪些层可以调整
            
            include_set={
                n
                for n, _ in self.model.named_parameters()
                if all(
                    e not in n.lower()
                    for e in (
                        "bias",
                        "norm",
                        "embeddings",
                        "classifier",
                        "pooler",
                        "shared",
                        "embed",
                        "positions",
                    )
                )
            },
            max_scale=self.hparams.max_scale,
            embedding_init=self.model.model.shared.weight.data,
        )
        # 我们的text2sql: embedding_init
        # model.encoder.embeddings.word_embeddings.weight.data


        self.alpha_kl = torch.nn.Parameter(torch.ones(()))
        self.alpha_kl.register_hook(lambda grad: -grad)

        self.alpha_lp = torch.nn.Parameter(torch.ones(()))
        self.alpha_lp.register_hook(lambda grad: -grad)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.valid_flipped = pl.metrics.Accuracy()

        self.register_buffer("margin_kl", torch.tensor(self.hparams.margin_kl_max))
        self.register_buffer("margin_lp", torch.tensor(self.hparams.margin_lp_max))
        self.running_flipped = []

    def train_dataloader(self, shuffle=True):
        if not hasattr(self, "train_dataset"):
            self.train_dataset = Seq2SeqAugmentedKILT(
                tokenizer=self.tokenizer,
                data_path=self.hparams.train_data_path,
                max_length=self.hparams.max_length,
                return_view=self.hparams.use_views,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )

    def val_dataloader(self, shuffle=True):
        if not hasattr(self, "val_dataset"):
            self.val_dataset = Seq2SeqAugmentedKILT(
                tokenizer=self.tokenizer,
                data_path=self.hparams.dev_data_path,
                max_length=self.hparams.max_length,
                return_view=self.hparams.use_views,
            )
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )

    def get_logits_orig_params_dict(self, batch):
        with torch.enable_grad():
            # 3 is batchsize,
            # ~~~src_input_ids~~~~ torch.Size([3, 16])
            # ~~~src_attention_mask~~~ torch.Size([3, 16])
            # ~~~trg_input_ids~~~~ torch.Size([3, 4])
            # ~~~trg_attention_mask~~~ torch.Size([3, 4])
            # ~~~~logits~~~ torch.Size([3, 4, 50265])
            # ~~~~~~~~logits_orig~~~~~~~ torch.Size([1, 4, 50265])
            # ~~~~~~~~~logit_for_grad~~~ torch.Size([1, 4, 50265])
            
            logits_orig, logit_for_grad, _ = self.model.eval()(
                input_ids=batch["src_input_ids"],
                attention_mask=batch["src_attention_mask"],
                decoder_input_ids=batch["trg_input_ids"][:, :-1],
                decoder_attention_mask=batch["trg_attention_mask"][:, :-1],
                use_cache=False,
            ).logits.split(
                [
                    len(batch["src_input_ids"]) - (2 if self.hparams.use_views else 1),
                    1,
                    1 if self.hparams.use_views else 0,
                ]
            )



            logits_orig = logits_orig.detach()
            # detach() 得到的tensor永远不需要计算其梯度，不具有grad。
            
            # 现在问题就是batch
            grads = torch.autograd.grad(
                label_smoothed_nll_loss(
                    logit_for_grad.log_softmax(-1),
                    batch["trg_input_ids"][
                        -2
                        if self.hparams.use_views
                        else -1 : -1
                        if self.hparams.use_views
                        else None,
                        1:,
                    ],
                    epsilon=self.hparams.eps,
                    ignore_index=self.tokenizer.pad_token_id,
                )[1]
                / batch["trg_attention_mask"][
                    -2
                    if self.hparams.use_views
                    else -1 : -1
                    if self.hparams.use_views
                    else None,
                    1:,
                ].sum(),
                self.model.parameters(),
            )
            grads = {
                name: grad
                for (name, _), grad in zip(self.model.named_parameters(), grads)
            }
        # params_dict 是 {name of layer : shift of parameter}
        params_dict = self.learner(
            batch["cond_input_ids"],
            batch["cond_attention_mask"],
            grads=grads,
        )


        return logits_orig, params_dict

    def forward(self, batch):
        logits_orig, params_dict = self.get_logits_orig_params_dict(batch)                       

        fmodel = make_functional(self.model).eval()

        logits = fmodel(
            input_ids=batch["src_input_ids"],
            attention_mask=batch["src_attention_mask"],
            decoder_input_ids=batch["trg_input_ids"][:, :-1],
            decoder_attention_mask=batch["trg_attention_mask"][:, :-1],
            use_cache=False,
            # 这里的params 是调整过后的，因为 params_dict.get(n, 0) 是shift of paramete， p是 parameter
            params=[
                params_dict.get(n, 0) + p for n, p in self.model.named_parameters()
            ],
        ).logits

        return logits_orig, logits, params_dict

    def get_kl_lp_cr(self, logits_orig, logits, labels, params_dict):
        # kl应该是constraint里的KL divergence
        kl = torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=logits_orig),
            torch.distributions.Categorical(
                logits=logits[: -2 if self.hparams.use_views else -1]
            ),
        )

        # lp应该是 lp norm 
        lp = sum(
            (p.abs() ** self.hparams.p).mean() ** (1 / self.hparams.p)
            for p in params_dict.values()
        ) / len(params_dict)
        # cr 不知道是什么
        cr, _ = label_smoothed_nll_loss(
            logits[-2 if self.hparams.use_views else -1 :].log_softmax(-1),
            labels[-2 if self.hparams.use_views else -1 :],
            epsilon=self.hparams.eps,
            ignore_index=self.tokenizer.pad_token_id,
        )

        return kl, lp, cr

    def training_step(self, batch, batch_idx):
        logits_orig, logits, params_dict = self.forward(batch)

        kl, lp, cr = self.get_kl_lp_cr(
            logits_orig, logits, batch["trg_input_ids"][:, 1:], params_dict
        )

        kl = (
            kl
            * batch["trg_attention_mask"][: (-2 if self.hparams.use_views else -1), 1:]
        ).sum() / batch["trg_attention_mask"][
            : (-2 if self.hparams.use_views else -1), 1:
        ].sum()

        cr = (
            cr
            / batch["trg_attention_mask"][
                (-2 if self.hparams.use_views else -1) :, 1:
            ].sum()
        )   

        loss_kl = self.alpha_kl * (kl - self.margin_kl)
        loss_lp = self.alpha_lp * (lp - self.margin_lp)

        if self.hparams.divergences == "both":
            loss = cr + loss_kl + loss_lp
        elif self.hparams.divergences == "kl":
            loss = cr + loss_kl
        elif self.hparams.divergences == "lp":
            loss = cr + loss_lp

        self.log("alpha_kl", self.alpha_kl, on_step=True, on_epoch=False, prog_bar=True)
        self.log("alpha_lp", self.alpha_lp, on_step=True, on_epoch=False, prog_bar=True)
        self.log("kl", kl, on_step=True, on_epoch=False, prog_bar=True)
        self.log("lp", lp, on_step=True, on_epoch=False, prog_bar=True)
        self.log("cr", cr, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx=None):

        _, params_dict = self.get_logits_orig_params_dict(batch)

        fmodel = make_functional(self.model).eval()

        gold = [b["pred"] for b in batch["raw"][:-1]] + [batch["raw"][-1]["alt"]] * (
            2 if self.hparams.use_views else 1
        )

        guess = self.tokenizer.batch_decode(
            fmodel.generate(
                input_ids=batch["src_input_ids"],
                attention_mask=batch["src_attention_mask"],
                min_length=0,
                num_beams=5,
                num_return_sequences=1,
                params=[
                    params_dict.get(n, 0) + p for n, p in self.model.named_parameters()
                ],
            ),
            skip_special_tokens=True,
        )

        acc = torch.tensor(
            [a.lower().strip() == b.lower().strip() for a, b in zip(guess, gold)]
        ).long()
        self.valid_acc(acc, torch.ones_like(acc))
        self.log(
            "valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        self.valid_flipped(
            acc[(-2 if self.hparams.use_views else -1) :],
            torch.ones_like(acc[(-2 if self.hparams.use_views else -1) :]),
        )
        self.log(
            "valid_flipped",
            self.valid_flipped,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def sample(
        self,
        sentences,
        condition,
        params_dict=None,
        num_return_sequences=1,
        stop_condition=None,
    ):
        len_sent = len(sentences)
        with torch.no_grad():
            batch = {
                k: v.to(self.device)
                for k, v in self.val_dataset.get_batch(sentences, condition).items()
            }

            if not params_dict:
                _, params_dict = self.get_logits_orig_params_dict(batch)

            fmodel = make_functional(self.model).eval()

            guess = list(
                batch_it(
                    self.tokenizer.batch_decode(
                        fmodel.generate(
                            input_ids=batch["src_input_ids"],
                            attention_mask=batch["src_attention_mask"],
                            min_length=0,
                            num_beams=5,
                            num_return_sequences=num_return_sequences,
                            params=[
                                params_dict.get(n, 0) + p
                                for n, p in self.model.named_parameters()
                            ],
                        ),
                        skip_special_tokens=True,
                    ),
                    num_return_sequences,
                )
            )

            n_iter = 1
            if stop_condition is not None and stop_condition(condition, guess, n_iter):
                model_tmp = deepcopy(self.model)
                params_dict_tmp = deepcopy(params_dict)

                while stop_condition(condition, guess, n_iter):
                    for n, p in self.model.named_parameters():
                        p.data += params_dict.get(n, 0)

                    guess = list(
                        batch_it(
                            self.tokenizer.batch_decode(
                                fmodel.generate(
                                    input_ids=batch["src_input_ids"],
                                    attention_mask=batch["src_attention_mask"],
                                    min_length=0,
                                    num_beams=5,
                                    num_return_sequences=num_return_sequences,
                                    params=[
                                        params_dict.get(n, 0) + p
                                        for n, p in self.model.named_parameters()
                                    ],
                                ),
                                skip_special_tokens=True,
                            ),
                            num_return_sequences,
                        )
                    )

                    params_dict_tmp = {
                        k: v + params_dict[k] for k, v in params_dict_tmp.items()
                    }
                    n_iter += 1

                self.model = model_tmp
                params_dict = params_dict_tmp

            if num_return_sequences == 1:
                guess = [e[0] for e in guess]

            return params_dict, guess[:len_sent]

    def on_before_zero_grad(self, optimizer):
        self.alpha_kl.data = torch.where(
            self.alpha_kl.data < 0,
            torch.full_like(self.alpha_kl.data, 0),
            self.alpha_kl.data,
        )
        self.alpha_lp.data = torch.where(
            self.alpha_lp.data < 0,
            torch.full_like(self.alpha_lp.data, 0),
            self.alpha_lp.data,
        )

    def on_validation_epoch_end(self):
        if self.valid_flipped.compute().item() > 0.9:
            self.margin_kl = max(
                self.margin_kl * 0.8, self.margin_kl * 0 + self.hparams.margin_kl_min
            )
            self.margin_lp = max(
                self.margin_lp * 0.8, self.margin_lp * 0 + self.hparams.margin_lp_min
            )
        self.log(
            "margin_kl", self.margin_kl, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "margin_lp", self.margin_lp, on_step=False, on_epoch=True, prog_bar=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            [
                {
                    "params": self.learner.parameters(),
                    "lr": self.hparams.lr,
                },
                {
                    "params": [self.alpha_kl, self.alpha_lp],
                    "lr": self.hparams.lr_alpha,
                },
            ],
            centered=True,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_updates,
            num_training_steps=self.hparams.total_num_updates,
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]




class Seq2SeqAugmentedKILT(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path,
        max_length=32,
        return_view=False,
        all_views=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        with jsonlines.open(data_path) as f:
            for d in f:
                self.data.append(
                    {
                        k: d[k]
                        for k in (
                            "input",
                            # 问题
                            "prediction",
                            # 模型输出（错误）
                            "alternatives",
                            #模型可能输出的其他答案
                            # 与input有一样prediction的语言相似的问题
                        )
                    }
                )

        self.max_length = max_length
        self.all_views = all_views
        self.return_view = return_view

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item, seed=None):
        alt = self.data[item]["alternatives"]
        output = {
            "src": self.data[item]["input"],
            "pred": self.data[item]["prediction"],
            "alt": alt,
            "cond": "{} >> {} || {}".format(
                self.data[item]["prediction"],
                alt,
                self.data[item]["input"],
            ),
        }

        return output

    def get_batch(self, sentences, condition):
        batch = {
            "{}_{}".format(k1, k2): v2
            for k1, v1 in {
                "src": sentences
                + [condition.split("|| ")[1]] * (1 + int(self.return_view)),
                "trg": [condition.split(" || ")[0].split(" >> ")[1]]
                * (len(sentences) + 1 + int(self.return_view)),
                "cond": [condition],
            }.items()
            for k2, v2 in self.tokenizer(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }
        batch["trg_input_ids"][:, 0] = self.tokenizer.eos_token_id
        return batch

    def collate_fn(self, batch):
        src = [b["src"] for b in batch]
        trg = [b["pred"] for b in batch[:-1]] + [batch[-1]["alt"]]

        if self.return_view:
            src += batch[-1]["view"] if self.all_views else [batch[-1]["view"]]
            trg += [batch[-1]["alt"]] * (
                len(batch[-1]["view"]) if self.all_views else 1
            )

        batches = {
            "{}_{}".format(k1, k2): v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "cond": [batch[-1]["cond"]],
            }.items()
            for k2, v2 in self.tokenizer(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["trg_input_ids"][:, 0] = self.tokenizer.eos_token_id
        batches["raw"] = batch
        return batches


class BartSeq2Seq(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--train_data_path",
            type=str,
            default="../datasets/structured_zeroshot-train-new.jsonl",
        )
        parser.add_argument(
            "--dev_data_path",
            type=str,
            default="../datasets/structured_zeroshot-dev-new.jsonl",
        )
        parser.add_argument("--batch_size", type=int, default=48)
        parser.add_argument("--lr", type=float, default=3e-5)
        parser.add_argument("--max_length", type=int, default=32)
        parser.add_argument("--weight_decay", type=int, default=0.01)
        parser.add_argument("--total_num_updates", type=int, default=50000)
        parser.add_argument("--warmup_updates", type=int, default=500)
        parser.add_argument("--num_workers", type=int, default=0)

        parser.add_argument("--model_name", type=str, default="t5-base")
        parser.add_argument("--eps", type=float, default=0.1)
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__()
        # self.save_hyperparameters()
        print('~~~~~~~~~~~~')
        print(self.hparams)
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.hparams.model_name
        )

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def train_dataloader(self, shuffle=True):
        if not hasattr(self, "train_dataset"):
            self.train_dataset = Seq2SeqKILT(
                tokenizer=self.tokenizer,
                data_path=self.hparams.train_data_path,
                max_length=self.hparams.max_length,
                templates=True,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            self.val_dataset = Seq2SeqKILT(
                tokenizer=self.tokenizer,
                data_path=self.hparams.dev_data_path,
                max_length=self.hparams.max_length,
                validation=True,
                templates=True,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=4,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def forward(self, batch):
        
        return self.model(
            input_ids=batch["src_input_ids"],
            attention_mask=batch["src_attention_mask"],
            decoder_input_ids=batch["trg_input_ids"][:, :-1],
            decoder_attention_mask=batch["trg_attention_mask"][:, :-1],
            use_cache=False,
        ).logits

    def training_step(self, batch, batch_idx=None):
        logits = self.forward(batch)

        loss, nll_loss = label_smoothed_nll_loss(
            logits.log_softmax(-1),
            batch["trg_input_ids"][:, 1:],
            epsilon=self.hparams.eps,
            ignore_index=self.tokenizer.pad_token_id,
        )

        ntokens = batch["trg_attention_mask"][:, 1:].sum()
        loss, nll_loss = loss / ntokens, nll_loss / ntokens

        self.log("nll_loss", nll_loss, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx=None):
        gold = [b["trg"] for b in batch["raw"]]
        guess = self.tokenizer.batch_decode(
            self.model.generate(
                input_ids=batch["src_input_ids"],
                attention_mask=batch["src_attention_mask"],
                min_length=0,
                num_beams=5,
                num_return_sequences=1,
            ),
            skip_special_tokens=True,
        )

        acc = torch.tensor(
            [
                a.lower().strip() in [c.lower().strip() for c in b]
                for a, b in zip(guess, gold)
            ]
        ).long()
        self.valid_acc(acc, torch.ones_like(acc))
        self.log(
            "valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True
        )

    def sample(self, sentences, num_return_sequences=1):
        self.eval()
        with torch.no_grad():
            return list(
                batch_it(
                    self.tokenizer.batch_decode(
                        self.model.generate(
                            **{
                                k: v.to(self.device)
                                for k, v in self.tokenizer(
                                    sentences,
                                    return_tensors="pt",
                                    padding=True,
                                    max_length=self.hparams.max_length,
                                    truncation=True,
                                ).items()
                            },
                            min_length=0,
                            num_beams=5,
                            num_return_sequences=num_return_sequences,
                        ),
                        skip_special_tokens=True,
                    ),
                    num_return_sequences,
                )
            )

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_updates,
            num_training_steps=self.hparams.total_num_updates,
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]



class ConditionedParameter(torch.nn.Module):
    # 这里应该是5个FFNN的代码
    def __init__(self, parameter, condition_dim=1024, hidden_dim=128, max_scale=1):
        super().__init__()
        self.parameter_shape = parameter.shape

        if len(self.parameter_shape) == 2:
            self.conditioners = torch.nn.Sequential(
                torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(
                        hidden_dim, 2 * (parameter.shape[0] + parameter.shape[1]) + 1
                    )
                ),
            )
        elif len(self.parameter_shape) == 1:
            self.conditioners = torch.nn.Sequential(
                torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(hidden_dim, 2 * parameter.shape[0] + 1)
                ),
            )
        else:
            raise RuntimeError()

        self.max_scale = max_scale

    def forward(self, inputs, grad):
        # 它这里写的跟论文里有出入，直接就是用一个 全链接层组成的module 暴力输出一个长vector， 然后再split成5个
        if len(self.parameter_shape) == 2:
            (
                conditioner_cola,
                conditioner_rowa,
                conditioner_colb,
                conditioner_rowb,
                conditioner_norm,
            ) = self.conditioners(inputs).split(
                [
                    self.parameter_shape[1],
                    self.parameter_shape[0],
                    self.parameter_shape[1],
                    self.parameter_shape[0],
                    1,
                ],
                dim=-1,
            )

            a = conditioner_rowa.softmax(-1).T @ conditioner_cola
            b = conditioner_rowb.softmax(-1).T @ conditioner_colb

        elif len(self.parameter_shape) == 1:
            a, b, conditioner_norm = self.conditioners(inputs).split(
                [self.parameter_shape[0], self.parameter_shape[0], 1], dim=-1
            )
        else:
            raise RuntimeError()
        # 这个是shift of parameter delta W
        return (
            self.max_scale
            * conditioner_norm.sigmoid().squeeze()
            * (grad * a.squeeze() + b.squeeze())
        )


class LSTMConditioner(torch.nn.Module):
    # 这一部分应该是LSTM + FFNN
    def __init__(
        self,
        vocab_dim=30522,
        embedding_dim=768,
        hidden_dim=256,
        #这里的hidden_dim也许是写错了，应该是128
        output_dim=1024,
        embedding_init=None,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_dim,
            embedding_dim=embedding_dim,
            padding_idx=0,
            _weight=embedding_init,
        )
        self.lstm = PytorchSeq2VecWrapper(
            torch.nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
        )
        self.linear = FeedForward(
            input_dim=hidden_dim * 2,
            num_layers=1,
            hidden_dims=[output_dim],
            activations=[torch.nn.Tanh()],
        )

    def forward(self, inputs, masks):
        return self.linear(self.lstm(self.embedding(inputs), masks))


class OneShotLearner(torch.nn.Module):
    def __init__(
        self,
        model,
        vocab_dim=30522,
        embedding_dim=768,
        hidden_dim=128,
        condition_dim=1024,
        include_set={},
        max_scale=1e-3,
        embedding_init=None,
    ):
        super().__init__()
        # {name: {name}_conditioner}
        self.param2conditioner_map = {
            n: "{}_conditioner".format(n).replace(".", "_")
            for n, p in model.named_parameters()
            if n in include_set
        }

        self.conditioners = torch.nn.ModuleDict(
            # {{name}_conditioner: 一个实例化的ConditionedParameter}
            {
                self.param2conditioner_map[n]: ConditionedParameter(
                    p,
                    condition_dim,
                    hidden_dim,
                    max_scale=max_scale,
                )
                for n, p in model.named_parameters()
                # include_set 说不定是可以控制的layer
                if n in include_set
            }
        )

        self.condition = LSTMConditioner(
            vocab_dim,
            embedding_dim,
            hidden_dim,
            condition_dim,
            embedding_init=embedding_init,
        )

    def forward(self, inputs, masks, grads=None):
        condition = self.condition(inputs, masks)
        return {
            #{name: shift of parameter}
            p: self.conditioners[self.param2conditioner_map[p]](
                # forward 时 LSTM的输出是ConditionedParameter的输入。
                condition,
                grad=grads[p] if grads else None,
            )
            for p, c in self.param2conditioner_map.items()
        }
