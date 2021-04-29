from yacs.config import CfgNode


def init_config():
    cfg = CfgNode({
        'exp_name': 'default',
        'model_name': 'laplace',
        'resume': True,
        'resume_ckpt': '', # load last checkpoint by default if ''
        'parallel': False,
        'device_ids': [0, 1],
        'device': 'cuda:0',
        'logdir': '../output/logs/',
        'checkpointdir': '../output/checkpoints/',

        'dataset': 'mnist',
        'dataset_paths': {
            'mnist': '../data/mnist',
        },

        # For engine.train
        'train': {
            'eval_on': False,
            'batch_size': 16,
            'max_epochs': 1000,
            'max_steps': 1000000,
            'learning_rate': 1e-5,

            'clip_norm': 1.0,
            'max_ckpt': 5,

            'print_every': 10000,
            'save_every': 100000,
            'eval_every': 100000,
        },

        'model': {
            'image_shape': (28, 28),
            'z_dim': 784,
            'recon_std': 1.0,
        }
    })
