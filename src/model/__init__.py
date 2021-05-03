from .vae import LaplaceVAE
from .vsc import VSC

__all__ = ['get_model']

def get_model(cfg):
    model = None
    if cfg.model_name == 'laplace':
        model = LaplaceVAE(cfg)
    if cfg.model_name == 'vsc':
        model = VSC(cfg)

    return model
