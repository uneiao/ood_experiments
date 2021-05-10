from .vae import LaplaceVAE, NormalVAE
from .vsc import VSC

__all__ = ['get_model']

def get_model(cfg):
    model = None
    if cfg.model_name == 'laplace':
        model = LaplaceVAE(cfg)
    if cfg.model_name == 'vsc':
        model = VSC(cfg)
    if cfg.model_name == 'vae':
        model = NormalVAE(cfg)

    return model
