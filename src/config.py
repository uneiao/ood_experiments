from yacs.config import CfgNode


def init_config():
    cfg = CfgNode({
        # global settings
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

        # For training/eval
        'train': {
            'eval_on': True,
            'batch_size': 512,
            'max_epochs': 10000,
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
        'val': {
            'batch_size': 512,
            'num_workers': 4,
        },

        # settings of models
        'vae': {
            'image_shape': (28, 28),
            'z_dim': 784,
            'recon_std': 0.1,
        },

        'vsc': {
            'image_shape': (28, 28),
            'z_dim': 784,
            'recon_std': 0.1,
            'tonolini_spike_c_start_step': 0,
            'tonolini_spike_c_end_step': 500000,
            'tonolini_spike_c_start_value': 1,
            'tonolini_spike_c_end_value': 10000,

            'prior_spike_prob': 0.1,
            'beta': 1,
        }

    })
    return cfg
