from .vae import LaplaceVAE

__all__ = ['get_model']

def get_model(cfg):
    model = None
    if cfg.model == 'LaplaceVAE':
        model = LaplaceVAE()

    return model
