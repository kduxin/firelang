import torch

class DummyOptimizer:
    def step(self): pass
    def zero_grad(self): pass


class DummyScheduler:
    def __init__(self, lr=0): self.lr = [lr]
    def step(self): pass
    def get_last_lr(self): return self.lr


class Loss:
    losses: dict
    def __init__(self, **kwargs):
        self.losses = kwargs
    
    def __getitem__(self, loss_name):
        return self.losses[loss_name]
    
    def __getattr__(self, loss_name):
        if loss_name in self.losses:
            return self.losses[loss_name]
        else:
            raise AttributeError(f'No loss {loss_name} found.')

    def add(self, name, loss):
        assert name not in self.losses
        self.losses[name] = loss
    
    def items(self):
        for key, val in self.losses.items():
            yield (key, val)

    def total(self):
        total = 0
        for key, val in self.items():
            total += val
        return total
    
    def reduced_items(self):
        for key, val in self.losses.items():
            yield (key, Loss._reduce(val))
    
    def reduced_total(self):
        total = 0
        for _, val in self.reduced_items():
            total += val
        return total
    
    def __iter__(self):
        return iter(self.losses.keys())
    
    def __repr__(self):
        return '<Loss({})>'.format(
            ','.join([f'{key}={val:.3g}' for key, val in self.reduced_items()])
        )
    
    @staticmethod
    def _reduce(tensor):
        assert isinstance(tensor, torch.Tensor)
        return tensor.mean()