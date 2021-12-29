import argparse
import itertools
import json
import os
import sys
import random
import operator

import _jsonnet
import torch
import tqdm
import attr


# These imports are needed for registry.lookup
# noinspection PyUnresolvedReferences
from text2qdmr.utils import registry
from text2qdmr.utils import saver as saver_mod
from text2qdmr.model.modules import decoder_utils
from text2qdmr.utils.serialization import ComplexEncoder
import math
from argparse import ArgumentParser
from copy import deepcopy
import pytorch_lightning as pl
import torch
from higher.patch import monkeypatch as make_functional
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from transformers import (
    BertModel,
    BertTokenizer,
    get_constant_schedule,
    get_linear_schedule_with_warmup,
    AutoTokenizer
)

from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pandas as pd 
from torch.utils.data import Dataset

@attr.s
class Hypothesis:
    inference_state = attr.ib()
    next_choices = attr.ib()
    score = attr.ib(default=0)

    choice_history = attr.ib(factory=list)
    score_history = attr.ib(factory=list)



class text2qdmrAugmentedKILT(Dataset):
    def __init__(
        self,
        preproc_data,
        orig_data,
        tokenizer,
        data_path,
        max_length=32,
        return_view=False,
        all_views=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.preproc_data = preproc_data
        self.orig_data = orig_data


        self.max_length = max_length
        self.all_views = all_views
        self.return_view = return_view

    def __len__(self):
        return len(self.preproc_data)

    def __getitem__(self, item, seed=None):
        output = {
            "src": self.preproc_data[item][0],
            "pred": self.preproc_data[item][2],
            "alt": self.preproc_data[item][1],
            "cond": "{} >> {} || {}".format(
                self.preproc_data[item][2].tree,
                self.preproc_data[item][1].tree,
                self.preproc_data[item][0]['raw_question'],
            ),
            "origin_data": self.orig_data[item]
        }

        return output


    def collate_fn(self, batch):
        batches = {}
        batches['src'] = [b["src"] for b in batch]
        batches['src_trg'] = [(b["src"],b["pred"]) for b in batch[:-1]] + [(batch[-1]["src"],batch[-1]["alt"])]
        batches['origin_data'] = [b["origin_data"] for b in batch]
        cond_input = self.tokenizer(
                [batch[-1]["cond"]],
                return_tensors="pt",
                padding=True,
                max_length=512,
                truncation=True,
            )


        batches['cond_input_ids'] = cond_input['input_ids']
        

        batches['cond_attention_mask'] = cond_input['attention_mask']
        

        return batches



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



class text2sqlAugmented(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--train_data_path",
            type=str,
            default="src/datasets/fever-train-kilt.jsonl",
        )
        parser.add_argument(
            "--dev_data_path",
            type=str,
            default="src/datasets/fever-dev-kilt.jsonl",
        )
        parser.add_argument("--batch_size", type=int, default=2)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--lr_alpha", type=float, default=1e-1)
        parser.add_argument("--max_length", type=int, default=512)
        parser.add_argument("--total_num_updates", type=int, default=200000)
        parser.add_argument("--warmup_updates", type=int, default=1000)
        parser.add_argument("--num_workers", type=int, default=3)

        parser.add_argument("--model_name", type=str, default="bert-base-uncased")
        parser.add_argument(
            "--model_checkpoint",
            type=str,
            default="src/models/FC_model.ckpt",
        )

        parser.add_argument("--margin_kl_max", type=float, default=1e-1)
        parser.add_argument("--margin_kl_min", type=float, default=1e-3)
        parser.add_argument("--margin_lp_max", type=float, default=1e-6)
        parser.add_argument("--margin_lp_min", type=float, default=1e-9)
        parser.add_argument("--max_scale", type=float, default=1)
        parser.add_argument("--p", type=float, default=2)
        parser.add_argument(
            "--divergences", type=str, choices=["kl", "lp", "both"], default="both"
        )
        parser.add_argument("--use_views", action="store_true")

        return parser

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained('Salesforce/grappa_large_jnt')
        # self.tokenizer = BertTokenizer.from_pretrained(self.hparams.model_name)
        self.kwargs = kwargs
        self.Infer, self.model, self.infer_config= get_model(self.kwargs)
        self.model.eval()
        self.section = 'val'
        self.output_history = False
        self.strict_decoding = True

        # self.model = BertBinary.load_from_checkpoint(
        #     self.hparams.model_checkpoint
        # ).model.eval()

        # self.adjust_part = 'rule_model'
        self.adjust_part = 'bert_rule_logits'

        if self.adjust_part == 'bert':
            # bert 模式有问题，没有对上logits，以及params dictkeys有问题
            self.learner = OneShotLearner(
                self.model.encoder,
                vocab_dim=self.model.encoder.bert_model.embeddings.word_embeddings.weight.data.shape[0],
                embedding_dim=self.model.encoder.bert_model.embeddings.word_embeddings.weight.data.shape[1],
                hidden_dim=128,
                condition_dim=1024,
                include_set={
                    n
                    for n, _ in self.model.encoder.named_parameters()
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
                            "encs_update.align_attn.linears.2.weight", # 这个会None，不知道为什么
                            "encs_update.align_attn.relation_v_emb.weight", # 这个会None，不知道为什么
                            "encs_update", # 算力有限，先只调整bert参数
                        )
                    )
                },
                max_scale=self.hparams.max_scale,
                embedding_init=self.model.encoder.bert_model.embeddings.word_embeddings.weight.data,
            )
        elif self.adjust_part == 'rule_model':

            self.learner = OneShotLearner(
                self.model.decoder.rule_logits,
                vocab_dim=self.model.encoder.bert_model.embeddings.word_embeddings.weight.data.shape[0],
                embedding_dim=self.model.encoder.bert_model.embeddings.word_embeddings.weight.data.shape[1],
                hidden_dim=128,
                condition_dim=1024,
                include_set={
                    n
                    for n, _ in self.model.decoder.rule_logits.named_parameters()
                    if all(
                        e not in n.lower()
                        for e in (
                            "embeddings",
                            "classifier",
                            "pooler",
                            "shared",
                            "embed",
                            "positions",
                            "encs_update.align_attn.linears.2.weight", # 这个会None，不知道为什么
                            "encs_update.align_attn.relation_v_emb.weight", # 这个会None，不知道为什么
                            "encs_update", # 算力有限，先只调整bert参数
                        )
                    )
                },
                max_scale=self.hparams.max_scale,
                embedding_init=self.model.encoder.bert_model.embeddings.word_embeddings.weight.data,
            )
        elif self.adjust_part == 'bert_rule_logits':
            self.learner = OneShotLearner(
                self.model.encoder.bert_model,
                vocab_dim=self.model.encoder.bert_model.embeddings.word_embeddings.weight.data.shape[0],
                embedding_dim=self.model.encoder.bert_model.embeddings.word_embeddings.weight.data.shape[1],
                hidden_dim=128,
                condition_dim=1024,
                include_set={
                    n
                    for n, _ in self.model.encoder.bert_model.named_parameters()
                    if any(
                        e in n.lower()
                        for e in (
                            ".layers.0.",
                            ".layer.0."   # 先试看第一层
                        )
                    )
                },
                max_scale=self.hparams.max_scale,
                embedding_init=self.model.encoder.bert_model.embeddings.word_embeddings.weight.data,
            )


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


    def _yield_batches_from_epochs(self, loader):
        while True:
            for batch in loader:
                yield batch

    def train_dataloader(self, shuffle=True):
        # TODO 这里的section可能需要修改

        if self.Infer.config.get('full_data') is not None and self.infer_config.part != 'spider':
            orig_data = registry.construct('dataset', self.Infer.config['full_data'][self.section])
        else:
            orig_data = self.infer_config.data

        preproc_data, filter_orig_data = self.Infer.model_preproc.dataset_KE(self.section, orig_data, two_datasets=self.Infer.config.get('full_data'))

        preproc_data.part = self.infer_config.part

        train_dataset = text2qdmrAugmentedKILT(
            preproc_data=preproc_data,
            orig_data=filter_orig_data,
            tokenizer=self.tokenizer,
            data_path=self.hparams.train_data_path,
            max_length=self.hparams.max_length,
            return_view=self.hparams.use_views,
        )
        
        return DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=train_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )

    def val_dataloader(self, shuffle=True):

        if self.Infer.config.get('full_data') is not None and self.infer_config.part != 'spider':
            orig_data = registry.construct('dataset', self.Infer.config['full_data'][self.section])
        else:
            orig_data = self.infer_config.data

        preproc_data, filter_orig_data = self.Infer.model_preproc.dataset_KE(self.section, orig_data, two_datasets=self.Infer.config.get('full_data'))

        preproc_data.part = self.infer_config.part

        val_dataset = text2qdmrAugmentedKILT(
            preproc_data=preproc_data,
            orig_data=filter_orig_data,
            tokenizer=self.tokenizer,
            data_path=self.hparams.train_data_path,
            max_length=self.hparams.max_length,
            return_view=self.hparams.use_views,
        )
        
        return DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=val_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )
                

    def get_logits_orig_params_dict(self, batch):
        with torch.enable_grad():

            if self.adjust_part == 'bert':

                enc_states, bert_output = self.model.encoder.eval()(batch['src'])
                logits_orig, logit_for_grad, _ =  bert_output.split(
                    [
                        len(batch["src"]) - (2 if self.hparams.use_views else 1),
                        1,
                        1 if self.hparams.use_views else 0,
                    ]
                )

                logits_orig = logits_orig.detach()


                grads = torch.autograd.grad(
                    self.model([batch['src_trg'][-1]]),
                    self.model.encoder.parameters(),
                    allow_unused=True
                )
                grads = {
                    name: grad for (name, _), grad in zip(self.model.encoder.named_parameters(), grads)
                }

            elif self.adjust_part == 'rule_model':

                losses_batched, rule_loss_batched, rule_logits = self.model.eval()(batch['src_trg'],ke=True)

                logits_orig, logit_for_grad, _ =  rule_logits.split(
                    [
                        len(batch["src"]) - (2 if self.hparams.use_views else 1),
                        1,
                        1 if self.hparams.use_views else 0,
                    ]
                )

                logits_orig = logits_orig.detach()

                # for all losses
                # grads = torch.autograd.grad(
                #     self.model([batch['src_trg'][-1]]),
                #     self.model.decoder.rule_logits.parameters(),
                #     allow_unused=True
                # )

                #  for rule loss only
                _ , rule_loss_batched, __ = self.model([batch['src_trg'][-1]], ke=True)
                
                grads = torch.autograd.grad(
                    rule_loss_batched,
                    self.model.decoder.rule_logits.parameters(),
                    allow_unused=True
                )

                grads = {
                    name: grad for (name, _), grad in zip(self.model.decoder.rule_logits.named_parameters(), grads)
                }

            elif self.adjust_part == 'bert_rule_logits':

                losses_batched, rule_loss_batched, rule_logits = self.model.eval()(batch['src_trg'],ke=True)

                logits_orig, logit_for_grad, _ =  rule_logits.split(
                    [
                        len(batch["src"]) - (2 if self.hparams.use_views else 1),
                        1,
                        1 if self.hparams.use_views else 0,
                    ]
                )

                logits_orig = logits_orig.detach()

                # for all losses
                # grads = torch.autograd.grad(
                #     self.model([batch['src_trg'][-1]]),
                #     self.model.decoder.rule_logits.parameters(),
                #     allow_unused=True
                # )

                #  for rule loss only
                _ , rule_loss_batched, __ = self.model([batch['src_trg'][-1]], ke=True)
                
                grads = torch.autograd.grad(
                    rule_loss_batched,
                    self.model.encoder.bert_model.parameters(),
                    allow_unused=True
                )


                grads = {
                    name: grad for (name, _), grad in zip(self.model.encoder.bert_model.named_parameters(), grads)
                }


        params_dict = self.learner(
            batch["cond_input_ids"],
            batch["cond_attention_mask"],
            grads=grads,
        )

        return logits_orig, params_dict

    def forward(self, batch, logits_orig=None, params_dict=None):
        if not params_dict:
            logits_orig, params_dict = self.get_logits_orig_params_dict(batch)


        if self.adjust_part == 'bert':
            _ , __, logits = self.model.eval()(batch['src_trg'], ke=True, bert_parameter=[params_dict.get(n, 0) + p for n, p in self.model.encoder.named_parameters()])
        elif self.adjust_part == 'rule_model':
            _ , __, logits = self.model.eval()(batch['src_trg'], ke=True, rule_parameter=[params_dict.get(n, 0) + p for n, p in self.model.decoder.rule_logits.named_parameters()])
        elif self.adjust_part == 'bert_rule_logits':
            _ , __, logits = self.model.eval()(batch['src_trg'], ke=True, bert_parameter=[params_dict.get(n, 0) + p for n, p in self.model.encoder.bert_model.named_parameters()])


        return logits_orig, logits, params_dict


    def get_kl_lp_cr(self, logits_orig, logits, labels, params_dict):
        # TODO  可能有问题看一下

        kl = torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=logits_orig),
            torch.distributions.Categorical(
                logits=logits[: -2 if self.hparams.use_views else -1]
            ),
        )

        # print(params_dict)

        # 暂时去掉lp

        # lp = sum(
        #     (p.abs() ** self.hparams.p).mean() ** (1 / self.hparams.p)
        #     for p in params_dict.values()
        # ) / len(params_dict)

        lp = 0 

        #  for all loss
        # cr = self.model([labels[-1]])

        _, cr, __ = self.model([labels[-1]] ,ke=True)


        return kl, lp, cr

    def training_step(self, batch, batch_idx=None):

        logits_orig, logits, params_dict = self.forward(batch)

        kl, lp , cr = self.get_kl_lp_cr(
            logits_orig, logits, batch["src_trg"], params_dict
        )
        kl = kl.mean()
        cr = cr.mean(-1)

        loss_kl = self.alpha_kl * (kl - self.margin_kl)
        loss_lp = 0 *self.alpha_lp * (lp - self.margin_lp)


        if self.hparams.divergences == "both":
            loss = cr + loss_kl + loss_lp
            
        elif self.hparams.divergences == "kl":
            loss = cr + loss_kl
        elif self.hparams.divergences == "lp":
            loss = cr + loss_lp

        self.log("alpha_kl", self.alpha_kl, on_step=True, on_epoch=False, prog_bar=True)
        self.log("alpha_lp", self.alpha_lp, on_step=True, on_epoch=False, prog_bar=True)
        self.log("kl", kl, on_step=True, on_epoch=False, prog_bar=True)
        # self.log("lp", lp, on_step=True, on_epoch=False, prog_bar=True)
        self.log("cr", cr, on_step=True, on_epoch=False, prog_bar=True)
        self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx=None):

        logits_orig, logits, params_dict = self.forward(batch)

        beam_size = 1

        gold = [b[1].tree for b in batch["src_trg"]]

        guess = []


        for orig_item, preproc_item in tqdm.tqdm(zip(batch['origin_data'], batch['src_trg']), total=len(batch['origin_data'])):
            assert orig_item.full_name == preproc_item[0]['full_name'], (orig_item.full_name, preproc_item[0]['full_name'])
            
            if self.adjust_part == 'bert':
                decoded = self.Infer._infer_one(self.model, orig_item, preproc_item, beam_size, self.output_history, self.strict_decoding, self.section, bert_parameter=[params_dict.get(n, 0) + p for n, p in self.model.encoder.named_parameters()])
            elif self.adjust_part == 'rule_model':
                decoded = self.Infer._infer_one(self.model, orig_item, preproc_item, beam_size, self.output_history, self.strict_decoding, self.section, rule_parameter=[params_dict.get(n, 0) + p for n, p in self.model.decoder.rule_logits.named_parameters()])
            elif self.adjust_part == 'bert_rule_logits':
                decoded = self.Infer._infer_one(self.model, orig_item, preproc_item, beam_size, self.output_history, self.strict_decoding, self.section, bert_parameter=[params_dict.get(n, 0) + p for n, p in self.model.encoder.bert_model.named_parameters()])

            tree = [d['model_output'] for d in decoded]

            guess += [tree]
        
        acc = torch.tensor(
            [str(a).lower().strip() == str(b).lower().strip() for a, b in zip(guess, gold)]
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
        logits_orig=None,
        params_dict=None,
        stop_condition=None,
    ):
        len_sent = len(sentences)
        with torch.no_grad():
            logits_orig, logits, params_dict = self.forward(
                {
                    k: v.to(self.device)
                    for k, v in self.val_dataset.get_batch(sentences, condition).items()
                },
                logits_orig=logits_orig,
                params_dict=params_dict,
            )

            n_iter = 1
            if stop_condition is not None and stop_condition(condition, logits, n_iter):
                model_tmp = deepcopy(self.model)
                params_dict_tmp = deepcopy(params_dict)

                while stop_condition(condition, logits, n_iter):
                    for n, p in self.model.named_parameters():
                        p.data += params_dict.get(n, 0)

                    _, logits, params_dict = self.forward(
                        {
                            k: v.to(self.device)
                            for k, v in self.val_dataset.get_batch(
                                sentences, condition
                            ).items()
                        }
                    )
                    params_dict_tmp = {
                        k: v + params_dict[k] for k, v in params_dict_tmp.items()
                    }
                    n_iter += 1

                self.model = model_tmp
                params_dict = params_dict_tmp

            return logits_orig, logits[:len_sent], params_dict

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




class Inferer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            torch.set_num_threads(1)
        # 0. Construct preprocessors
        self.model_preproc = registry.instantiate(
            registry.lookup('model', config['model']).Preproc,
            config['model'])
        self.model_preproc.load()

    def load_model(self, logdir, step):
        '''Load a model (identified by the config used for construction) and return it'''
        # 1. Construct model
        model = registry.construct('model', self.config['model'], preproc=self.model_preproc, device=self.device)
        model.to(self.device)
        model.eval()


        # 2. Restore its parameters
        saver = saver_mod.Saver({"model": model})
        last_step = saver.restore(logdir, step=step, map_location=self.device, item_keys=["model"])
        if not last_step:
            raise Exception(f"Attempting to infer on untrained model in {logdir}, step={step}")
        return model

    def infer(self, model, output_path, args):
        output = open(output_path, 'w')
        
        with torch.no_grad():
            for section in args.section:
                if self.config.get('full_data') is not None and args.part != 'spider':
                    orig_data = registry.construct('dataset', self.config['full_data'][section])
                else:
                    orig_data = args.data[section]

                preproc_data = self.model_preproc.dataset(section, two_datasets=self.config.get('full_data'))

                preproc_data.part = args.part

                if args.shuffle:
                    idx_shuffle = list(range(len(orig_data)))
                    random.shuffle(idx_shuffle)
                    if args.limit:
                        idx_shuffle = idx_shuffle[:args.limit]
                    sliced_orig_data, sliced_preproc_data = [], []
                    for i, (orig_item, preproc_item) in enumerate(zip(orig_data, preproc_data)):
                        if i in idx_shuffle:
                            sliced_orig_data.append(orig_item)
                            sliced_preproc_data.append(preproc_item)
                else:
                    if args.limit:
                        sliced_orig_data = list(itertools.islice(orig_data, args.limit))
                        sliced_preproc_data = list(itertools.islice(preproc_data, args.limit))
                    else:
                        sliced_orig_data = orig_data
                        sliced_preproc_data = preproc_data
                self._inner_infer(model, args.beam_size, args.output_history, sliced_orig_data, sliced_preproc_data,
                                    output, args.strict_decoding, section)

    def _inner_infer(self, model, beam_size, output_history, sliced_orig_data, sliced_preproc_data, output, \
                    strict_decoding=False, section='val'):
        for orig_item, preproc_item in tqdm.tqdm(zip(sliced_orig_data, sliced_preproc_data), total=len(sliced_orig_data)):
            assert orig_item.full_name == preproc_item[0]['full_name'], (orig_item.full_name, preproc_item[0]['full_name'])
            
            decoded = self._infer_one(model, orig_item, preproc_item, beam_size, output_history, strict_decoding, section)
            
            # try:
            #     decoded = self._infer_one(model, orig_item, preproc_item, beam_size, output_history, strict_decoding, section)
            # except:
            #     print("skip:" , orig_item.full_name)
            #     continue


            
            # inputneed = open("logdir/grappa_qdmr_train_aug/input.infer", 'w')

            # inputneed.write(
            #     json.dumps({
            #         'orig_item': str(orig_item),
            #         'preproc_item': str(preproc_item),
            #         'output_history': str(output_history),
            #         'strict_decoding': str(strict_decoding),
            #         'section': str(section),
            #         'decoded': str(decoded),

            #     }, cls=ComplexEncoder) + '\n')
            # inputneed.flush()       

            output.write(
                json.dumps({
                    'name': orig_item.full_name,
                    'part': section,
                    'beams': decoded,
                }, cls=ComplexEncoder) + '\n')
            output.flush()

    def init_decoder_infer(self, model, data_item, section, strict_decoding):
        # model.decoder -> qdmr_dec.py BreakDecoder

        # schema
        model.decoder.schema = data_item.schema
        # grounding choices
        _, validation_info = model.preproc.validate_item(data_item, section)
        model.decoder.value_unit_dict = validation_info[0]
        model.decoder.ids_to_grounding_choices = model.decoder.preproc.grammar.get_ids_to_grounding_choices(data_item.schema, validation_info[0])
        for rule, idx in model.decoder.rules_index.items():
            if rule[1] == 'NextStepSelect':
                model.decoder.select_index = idx

        if strict_decoding:
            # column types
            assert len(data_item.column_data) == 1
            model.decoder.column_data = data_item.column_data[0]

            # info about value set
            model.decoder.no_vals = len(model.decoder.value_unit_dict) == 0

            model.decoder.required_column = False
            if not model.decoder.no_vals:
                model.decoder.no_column, model.decoder.required_column = True, True
                for val_units in model.decoder.value_unit_dict.values():
                    model.decoder.required_column = model.decoder.required_column and all(val_unit.column for val_unit in val_units)

            model.decoder.value_columns = set()
            model.decoder.val_types_wo_cols = set()
            for grnd_choice in model.decoder.ids_to_grounding_choices.values():
                if grnd_choice.choice_type == 'value':
                    for val_unit in grnd_choice.choice:
                        if val_unit.column:
                            model.decoder.value_columns.add((val_unit.table, val_unit.column))
                        else:
                            model.decoder.val_types_wo_cols.add(val_unit.value_type)
            for table in model.decoder.column_data.keys():
                for column, col_type in model.decoder.column_data[table].items():
                    if col_type in model.decoder.val_types_wo_cols:
                        model.decoder.value_columns.add((table, column))

            model.decoder.no_column = len(model.decoder.value_columns) == 0
        else:
            model.decoder.column_data = None

        return model

    def _infer_one(self, model, data_item, preproc_item, beam_size, output_history=False, strict_decoding=False, section='val',bert_parameter=None, rule_parameter=None):
        #data_item is for one question

        # data_item
        # dict_keys(['subset_idx', 'text', 'text_toks', 'text_toks_for_val', 'qdmr_code', 'qdmr_ops', 'qdmr_args', 'grounding', 'values', 'column_data', 'sql_code', 'schema', 'eval_graphs', 'orig_schema', 'orig_spider_entry', 'db_id', 'subset_name', 'full_name'])
        
        model = self.init_decoder_infer(model, data_item, section, strict_decoding)
        
        # preproc_item is the input of entire model
        # it includes :
        # raw question
        # tokenized question
        # schema linking
        # database id
        # column
        # values
        # general_grounding
        # general_grounding_types
        # BreakDecoderPreprocItem

        #这以下都需要
        beams = decoder_utils.beam_search(
                model, preproc_item, beam_size=beam_size, max_steps=1000, strict_decoding=strict_decoding,bert_parameter=bert_parameter, rule_parameter=rule_parameter)
        # only one in beams

        decoded = []
        # 以上都没有扯到rule model的parameters

        for beam in beams: # len(beams) = 1
            model_output, inferred_code = beam.inference_state.finalize()
            # 一个 model_output 是tree 结构 qdmr
            # inferred_code， 正常结构的qdmr
            decoded.append({
                'orig_question': data_item.text, #输入的问题
                'model_output': model_output, #action sequence
                'inferred_code': inferred_code, # qdmr
                'score': beam.score,
                **({
                       'choice_history': beam.choice_history,
                       'score_history': beam.score_history,
                   } if output_history else {})})
        return decoded


@attr.s
class InferConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    beam_size = attr.ib()
    output = attr.ib()
    step = attr.ib()
    strict_decoding = attr.ib(default=False)
    mode = attr.ib(default="infer")
    limit = attr.ib(default=None)
    part = attr.ib(default='spider')
    shuffle = attr.ib(default=False)
    output_history = attr.ib(default=False)
    data = attr.ib(default=None)


def main(args):

    exp_config = json.loads(_jsonnet.evaluate_file(args.exp_config_file))
    model_config_file = exp_config["model_config"]
    if "model_config_args" in exp_config:
        model_config_args = exp_config["model_config_args"]
        if args.model_config_args is not None:
            model_config_args_json = _jsonnet.evaluate_snippet("", args.model_config_args)
            model_config_args.update(json.loads(model_config_args_json))
        model_config_args = json.dumps(model_config_args)
    elif args.model_config_args is not None:
        model_config_args = _jsonnet.evaluate_snippet("", args.model_config_args)
    else:
        model_config_args = None

    logdir = args.logdir or exp_config["logdir"]
    name = exp_config["name"]


    if model_config_args:
        config = json.loads(_jsonnet.evaluate_file(model_config_file, tla_codes={'args': model_config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(model_config_file))
        
    model_preproc = registry.instantiate(
        registry.lookup('model', config['model']).Preproc,
        config['model'])
    data = {}
    for section in exp_config["eval_section"]:
        print('Load dataset, {} part'.format(section))
        orig_data = registry.construct('dataset', config['data'][section])
        orig_data.examples = model_preproc.load_raw_dataset(section, paths=config['data'][section]['paths'])
        orig_data.examples_with_name = {ex.full_name: ex for ex in orig_data.examples}
        data[section] = orig_data

    infer_output_path = f"{exp_config['eval_output']}/{exp_config['eval_name']}-step82000.infer"


    infer_config = InferConfig(
        model_config_file,
        model_config_args,
        logdir,
        exp_config["eval_section"],
        exp_config["eval_beam_size"],
        infer_output_path,
        82000,
        strict_decoding=exp_config.get("eval_strict_decoding", False),
        limit=exp_config.get("limit", None),
        shuffle=exp_config.get("shuffle", False),
        part=exp_config.get("part", 'spider'),
        data=data,
        )



    if model_config_args:
        config = json.loads(_jsonnet.evaluate_file(model_config_file, tla_codes={'args': model_config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(model_config_file))

    if 'model_name' in config:
        logdir = os.path.join(logdir, config['model_name'])

    output_path = infer_output_path.replace('__LOGDIR__', logdir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        print(f'Output file {output_path} already exists')
        sys.exit(1)

    inferer = Inferer(config)
    model = inferer.load_model(logdir, 82000)
    inferer.infer(model, output_path, infer_config)


def get_model(args):
    exp_config = json.loads(_jsonnet.evaluate_file(args['exp_config_file']))
    model_config_file = exp_config["model_config"]
    if "model_config_args" in exp_config:
        model_config_args = exp_config["model_config_args"]
        if args['model_config_args'] is not None:
            model_config_args_json = _jsonnet.evaluate_snippet("", args['model_config_args'])
            model_config_args.update(json.loads(model_config_args_json))
        model_config_args = json.dumps(model_config_args)
    elif args['model_config_args'] is not None:
        model_config_args = _jsonnet.evaluate_snippet("", args['model_config_args'])
    else:
        model_config_args = None

    logdir = args['logdir'] or exp_config["logdir"]
    name = exp_config["name"]


    if model_config_args:
        config = json.loads(_jsonnet.evaluate_file(model_config_file, tla_codes={'args': model_config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(model_config_file))
        
    model_preproc = registry.instantiate(
        registry.lookup('model', config['model']).Preproc,
        config['model'])
    data = {}
    for section in exp_config["eval_section"]:
        print('Load dataset, {} part'.format(section))
        orig_data = registry.construct('dataset', config['data'][section])
        orig_data.examples = model_preproc.load_raw_dataset(section, paths=config['data'][section]['paths'])
        orig_data.examples_with_name = {ex.full_name: ex for ex in orig_data.examples}
        data[section] = orig_data

    infer_output_path = f"{exp_config['eval_output']}/{exp_config['eval_name']}-step82000.infer"


    infer_config = InferConfig(
        model_config_file,
        model_config_args,
        logdir,
        exp_config["eval_section"],
        exp_config["eval_beam_size"],
        infer_output_path,
        82000,
        strict_decoding=exp_config.get("eval_strict_decoding", False),
        limit=exp_config.get("limit", None),
        shuffle=exp_config.get("shuffle", False),
        part=exp_config.get("part", 'spider'),
        data=data,
        )



    if model_config_args:
        config = json.loads(_jsonnet.evaluate_file(model_config_file, tla_codes={'args': model_config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(model_config_file))

    if 'model_name' in config:
        logdir = os.path.join(logdir, config['model_name'])

    output_path = infer_output_path.replace('__LOGDIR__', logdir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        print(f'Output file {output_path} already exists')
        sys.exit(1)

    inferer = Inferer(config)
    model = inferer.load_model(logdir, 82000)

    return inferer, model, infer_config



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help="preprocess/preprocess-dev/train/eval/eval-wo-infer", default= "eval")
    parser.add_argument('--exp_config_file', help="jsonnet file for experiments", default = "text2qdmr/configs/experiments/grappa_qdmr_train_aug.jsonnet")
    parser.add_argument('--model_config_args', help="optional overrides for model config args")
    parser.add_argument('--logdir', help="optional override for logdir")
    parser.add_argument('--backend_ditributed', type=str, default="nccl", help="backend to pass into torch.distributed.init_process_group")
    parser.add_argument('--partition', help="optional choice of partition (for preprocess)")
    parser.add_argument(
        "--dirpath", type=str, default="models/bert_binary_augmented_fever"
    )
    parser.add_argument("--save_top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser = text2sqlAugmented.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args, _ = parser.parse_known_args()

    seed_everything(seed=args.seed)

    logger = TensorBoardLogger(args.dirpath, name=None)

    callbacks = [
        ModelCheckpoint(
            monitor="valid_acc",
            mode="max",
            dirpath=os.path.join(logger.log_dir, "checkpoints"),
            save_top_k=args.save_top_k,
            filename="model-{epoch:02d}-{valid_acc:.4f}-{valid_flipped:4f}",
        ),
        LearningRateMonitor(
            logging_interval="step",
        ),
    ]

    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks)

    augmented_model  = text2sqlAugmented(**vars(args))

    trainer.fit(augmented_model)



    # main(args)
