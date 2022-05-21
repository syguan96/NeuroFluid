import os
import json
import argparse
import os.path as osp
from collections import namedtuple

class BaseTrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        io = self.parser.add_argument_group('path related variables')
        io.add_argument('--expdir', type=str, default='exps-simulation', help='experiment dir')
        io.add_argument('--expname', type=str, default='debug', help='experiment name')
        io.add_argument('--from_json', default=None, help='Load options from json file')
        io.add_argument('--resume_from', default=None, help='path of ckpt to be resumed')
        
        self.init_options()
    
    def init_options(self):
        pass

    def parse_args(self):
        self.args = self.parser.parse_args()
        if self.args.from_json is not None:
            path_to_json = os.path.abspath(self.args.from_json)
            with open(path_to_json, "r") as f:
                json_args = json.load(f)
                json_args = namedtuple("json_args", json_args.keys())(**json_args)
                return json_args
        else:
            self.save_dump()
            return self.args

    def save_dump(self):
        exp_path = osp.join(self.args.expdir, self.args.expname)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        with open(os.path.join(exp_path, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        return