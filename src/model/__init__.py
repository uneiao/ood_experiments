from .vae import LaplaceVAE, NormalVAE
from .vsc import VSC
from .mathieu19a_vae import SparseVAE
from .deform_vae import DeformVAE

__all__ = ['get_model']

def get_model(cfg):
    model = None
    if cfg.model_name == 'laplace':
        model = LaplaceVAE(cfg)
    if cfg.model_name == 'vsc':
        model = VSC(cfg)
    if cfg.model_name == 'vae':
        model = NormalVAE(cfg)
    if cfg.model_name == 'mathieu':
        model = SparseVAE(cfg)
    if cfg.model_name == 'deform_vae':
        model = DeformVAE(cfg)

    return model
