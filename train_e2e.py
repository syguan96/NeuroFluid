"""
training the whole framework.
"""

import torch
from trainer.trainer_e2e import Trainer
from configs import dataset_config, end2end_training_config

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    cfg_datasets = dataset_config()
    cfg_e2e = end2end_training_config()

    cfg_dataset = cfg_datasets[cfg_e2e.dataset]
    cfg_e2e.update(cfg_dataset)
    print(cfg_e2e.dump())

    trainer = Trainer(cfg_e2e)
    trainer.train()
