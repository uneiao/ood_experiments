from yacs.config import CfgNode


def init_config():
    cfg = CfgNode({
        'seed': 1,
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
        'dataset_path': {
            'mnist': '../data/mnist',
        },

        # For engine.train
        'train': {
            'eval_on': False,
            'batch_size': 512,
            'max_epochs': 5000,
            'max_steps': 10000000,
            'learning_rate': 1e-5,

            'num_workers': 4,

            'clip_norm': 1.0,
            'max_ckpt': 5,

            'solver': {
                'optim': 'Adam',
                'lr': 1e-5,
            },

            'print_every': 1000,
            'save_every': 10000,
            'eval_every': 10000,
        },

        'model': {
            'image_shape': (28, 28),
            'z_dim': 784,
            'recon_std': 0.1,
        }
    })
    return cfg