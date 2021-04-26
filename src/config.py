import yacs


def init_config():
    cfg = yacs.config.CfgNode({
        'resume': True,
        'resume_ckpt': '', # load last checkpoint by default if ''
        'parallel': False,
        'device_ids': [0, 1],
        'device': 'cuda:0',
        'logdir': '../output/logs/',
        'checkpointdir': '../output/checkpoints/',

        'dataset': '',
        'dataset_paths': {
            'mnist': '../data/mnist',
        },

            # For engine.train
        'train': {
            'batch_size': 16,
            'max_epochs': 1000,
            'max_steps': 1000000,
            'learning_rate': 1e-5,

            'clip_norm': 1.0,
            'max_ckpt': 5,
        },
    })
