"""
warmup renderer
"""

import torch

# from options.renderer_options_train import TrainOptions
from trainer.trainer_renderer import Trainer
from configs import dataset_config, warmup_training_config

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # options = TrainOptions().parse_args()
    cfg_datasets = dataset_config()
    cfg_warmup = warmup_training_config()

    cfg_dataset = cfg_datasets[cfg_warmup.dataset]
    cfg_warmup.update(cfg_dataset)
    print(cfg_warmup.dump())
    trainer = Trainer(cfg_warmup)
    trainer.train()
