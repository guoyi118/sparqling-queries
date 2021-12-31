from flask_restful import Resource, reqparse
import os
import sys 
import argparse
import json
import _jsonnet
import attr
from text2qdmr.commands import preprocess, train, infer, eval, forapi
from text2qdmr.utils import registry

import torch

@attr.s
class forapiConfig:
    question = attr.ib()
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





class Service(Resource):
    
    def post(self):

        parser = reqparse.RequestParser()
        parser.add_argument('question', required=True, help="test cannot be blank!")
        parser.add_argument('mode', help="preprocess/preprocess-dev/train/eval/eval-wo-infer", default="oneshot")
        parser.add_argument('exp_config_file', help="jsonnet file for experiments", default="text2qdmr/configs/experiments/grappa_qdmr_oneshot.jsonnet")
        parser.add_argument('model_config_args', help="optional overrides for model config args")
        parser.add_argument('logdir', help="optional override for logdir")
        parser.add_argument('backend_ditributed', type=str, default="nccl", help="backend to pass into torch.distributed.init_process_group")
        parser.add_argument('partition', help="optional choice of partition (for preprocess)")
        args = parser.parse_args()

    # def synchronize(self):
    #     """
    #     Helper function to synchronize (barrier) among all processes when
    #     using distributed training
    #     """
        # if not torch.distributed.is_available():
        #     return
        # if not torch.distributed.is_initialized():
        #     return
        # world_size = torch.distributed.get_world_size()
        # if world_size == 1:
        #     return
        # torch.distributed.barrier()

        num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        args.distributed = num_gpus > 1

        if args.distributed:
            assert "LOCAL_RANK" in os.environ
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(backend=args.backend_ditributed)
            # self.synchronize()
            print("Running with {} GPUs, this is GPU {}".format(num_gpus, local_rank))
        else:
            print("Running with 1 GPU")

        
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

        step = 81000
        infer_output_path = f"{exp_config['eval_output']}/{exp_config['eval_name']}-step{step}.infer"
        
        if args.mode == "oneshot":
            forapi_config = forapiConfig(
                args.question,
                model_config_file,
                model_config_args,
                logdir,
                exp_config["eval_section"],
                exp_config["eval_beam_size"],
                infer_output_path,
                step,
                strict_decoding=exp_config.get("eval_strict_decoding", False),
                limit=exp_config.get("limit", None),
                shuffle=exp_config.get("shuffle", False),
                part=exp_config.get("part", 'spider'),
                data=data,
            )
            
            result = forapi.main(forapi_config)
            
            return  result