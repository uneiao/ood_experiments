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
        'evaldir': '../output/eval/',

        'dataset': 'mnist',
        'dataset_path': {
            'mnist': '../data/mnist',
        },
        'mnist': {
            'in_class': [0, 1, 2, 3, 4, ],
            'total_num_class': 10,
        },

        # For training/eval
        'train': {
            'eval_on': True,
            'batch_size': 512,
            'max_epochs': 20000,
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
        'eval': {
            'batch_size': 512,
            'num_workers': 4,

            'checkpoint': 'last',
        },

        # settings of models
        'vae': {
            'image_shape': (28, 28),
            'z_dim': 784,
            'recon_std': 0.1,
            'prior_std': 1.0,
        },

        'vsc': {
            'image_shape': (28, 28),
            'z_dim': 64,
            'recon_std': 1.0,
            'tonolini_spike_c_start_step': 20000,
            'tonolini_spike_c_end_step': 160000,
            'tonolini_spike_c_start_value': 50,
            'tonolini_spike_c_end_value': 1000,

            'tonolini_lambda_start_step': 160000,
            'tonolini_lambda_end_step': 200000,
            'tonolini_lambda_start_value': 0.0,
            'tonolini_lambda_end_value': 1.0,

            'prior_spike_prob': 0.1,
            'beta': 1,
        },

        'mathieu': {
            'image_shape': (28, 28),
            'z_dim': 64,
            'recon_std': 1.0,
            'prior_std': 1.0,

            # sparse prior
            'gamma': 0.8,
            'loc':  0.0,
            'scale': 1.0,
            'spike_scale': 0.05,

            'beta': 1.0,
            'alpha': 1.0,
        },

    })
    return cfg
