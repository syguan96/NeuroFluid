"""
training the transition model
"""

import torch

from trainer.trainer_transmodel import Trainer
from configs import transmodel_config

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    options = transmodel_config()
    print(options.dump())
    trainer = Trainer(options)
    trainer.train()
