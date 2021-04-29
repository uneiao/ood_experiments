from torch.optim import Adam, RMSprop

__all__ = ['get_optimizer']

def get_optimizer(cfg, model):
    optim_class = {
        'Adam': Adam,
        'RMSprop': RMSprop
    }[cfg.train.solver.optim]

    return optim_class(model.parameters(), lr=cfg.train.solver.lr)
