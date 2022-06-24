"""
Config
"""

import os
import argparse
from typing import Dict
from yacs.config import CfgNode as CN

parser = argparse.ArgumentParser()
parser.add_argument('--expdir', type=str, default='exps', help='experiment dir')
parser.add_argument('--expname', type=str, default='debug', help='experiment name')
parser.add_argument('--dataset', type=str, default='', help='dataset')
parser.add_argument('--config', type=str, default='', help='default config file')
parser.add_argument('--resume_from', type=str, default='', help='path of ckpt to be load')

_C = CN(new_allowed=True)

def _parse_args(parser) -> Dict:
    args = parser.parse_args()
    return vars(args)


def default_config() -> CN:
    """
    Get a yacs CfgNode object with the default config values.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def get_config(config_file: str, merge: bool = True) -> CN:
    """
    Read a config file and optionally merge it with the default config file.
    Args:
      config_file (str): Path to config file.
      merge (bool): Whether to merge with the default config or not.
    Returns:
      CfgNode: Config as a yacs CfgNode object.
    """
    if merge:
      cfg = default_config()
    else:
      cfg = CN(new_allowed=True)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


def save_config(cfg, savepath):
    with open(savepath, 'w') as f:
        f.write(cfg.dump())


def dataset_config() -> CN:
    """
    Get dataset config file
    Returns:
      CfgNode: dataset config as a yacs CfgNode object.
    """
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset.yaml')
    cfg = _C.clone()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


def end2end_training_config() -> CN:
    """
    Get end2end training config file
    Returns:
      CfgNode: end2end traning config as a yacs CfgNode object.
    """
    cfg_argparse = _parse_args(parser)
    if cfg_argparse['config'] == '':
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'end2end.yaml')
    else:
        config_file = cfg_argparse['config']
    cfg = default_config()
    cfg.merge_from_file(config_file)
    cfg.update(cfg_argparse)
    cfg.freeze()

    os.makedirs(os.path.join(cfg_argparse['expdir'], cfg_argparse['expname']))
    _savepath = os.path.join(cfg_argparse['expdir'], cfg_argparse['expname'], 'config.yaml')
    save_config(cfg, _savepath)

    return cfg

def nerf_training_config() -> CN:
    """
    Get NeRF training config file
    Returns:
      CfgNode: NeRF traning config as a yacs CfgNode object.
    """
    cfg_argparse = _parse_args(parser)
    if cfg_argparse['config'] == '':
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nerf.yaml')
    else:
        config_file = cfg_argparse['config']
    cfg = default_config()
    cfg.merge_from_file(config_file)
    cfg.update(cfg_argparse)
    cfg.freeze()
    # print(cfg.dump())

    os.makedirs(os.path.join(cfg_argparse['expdir'], cfg_argparse['expname']))
    _savepath = os.path.join(cfg_argparse['expdir'], cfg_argparse['expname'], 'config.yaml')
    save_config(cfg, _savepath)

    return cfg

def warmup_training_config() -> CN:
    """
    Get warmup training config file
    Returns:
      CfgNode: warmup traning config as a yacs CfgNode object.
    """
    cfg_argparse = _parse_args(parser)
    if cfg_argparse['config'] == '':
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'warmup.yaml')
    else:
        config_file = cfg_argparse['config']
    cfg = default_config()
    cfg.merge_from_file(config_file)
    cfg.update(cfg_argparse)
    cfg.freeze()
    # print(cfg.dump())

    os.makedirs(os.path.join(cfg_argparse['expdir'], cfg_argparse['expname']))
    _savepath = os.path.join(cfg_argparse['expdir'], cfg_argparse['expname'], 'config.yaml')
    save_config(cfg, _savepath)

    return cfg

    
def transmodel_config() -> CN:
    """
    Get transmodel config file
    Returns:
      CfgNode: transmodel config as a yacs CfgNode object.
    """
    cfg_argparse = _parse_args(parser)
    if cfg_argparse['config'] == '':
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'transmodel.yaml')
    else:
        config_file = cfg_argparse['config']
    cfg = default_config()
    cfg.merge_from_file(config_file)
    cfg.update(cfg_argparse)
    cfg.freeze()
    # print(cfg.dump())

    os.makedirs(os.path.join(cfg_argparse['expdir'], cfg_argparse['expname']))
    _savepath = os.path.join(cfg_argparse['expdir'], cfg_argparse['expname'], 'config.yaml')
    save_config(cfg, _savepath)

    return cfg