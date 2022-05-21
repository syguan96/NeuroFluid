import torch

class ExponentialLR(torch.optim.lr_scheduler._LRScheduler):
    
    def __init__(self, optimizer, decay_epochs, gamma=0.1, last_epoch=-1):
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch / self.decay_epochs)
                for base_lr in self.base_lrs]

class WarmupExponentialLR(torch.optim.lr_scheduler._LRScheduler):
    
    def __init__(self, optimizer, decay_epochs, warmup_epochs=10000, gamma=0.1, last_epoch=-1):
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        super(WarmupExponentialLR, self).__init__(optimizer, last_epoch)
    
    def warmup(self):
        
        return float(self.last_epoch+1) / float(self.warmup_epochs)
    
    def exp_decay(self):
        return self.gamma ** (self.last_epoch / self.decay_epochs)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * self.warmup()
                for base_lr in self.base_lrs]
        else: 
            return [base_lr * self.exp_decay()
                for base_lr in self.base_lrs]