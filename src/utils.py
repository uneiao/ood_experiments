from collections import defaultdict, deque
from datetime import datetime
import json
import math
import os
import os.path as osp
import pickle

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Checkpointer:
    def __init__(self, checkpointdir, max_num):
        self.max_num = max_num
        self.checkpointdir = checkpointdir
        if not osp.exists(checkpointdir):
            os.makedirs(checkpointdir)
        self.listfile = osp.join(checkpointdir, 'model_list.pkl')

        if not osp.exists(self.listfile):
            with open(self.listfile, 'wb') as f:
                model_list = []
                pickle.dump(model_list, f)

    def save(self, path: str, model, optimizer, epoch, global_step):
        assert path.endswith('.pth')
        os.makedirs(osp.dirname(path), exist_ok=True)

        if isinstance(model, nn.DataParallel):
            model = model.module
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'epoch': epoch,
            'global_step': global_step
        }
        with open(path, 'wb') as f:
            torch.save(checkpoint, f)
            print('Checkpoint has been saved to "{}".'.format(path))

    def save_last(self, model, optimizer, epoch, global_step):
        path = osp.join(self.checkpointdir, 'model_{:09}.pth'.format(global_step + 1))

        with open(self.listfile, 'rb+') as f:
            model_list = pickle.load(f)
            if len(model_list) >= self.max_num:
                if osp.exists(model_list[0]):
                    os.remove(model_list[0])
                del model_list[0]
            model_list.append(path)
        with open(self.listfile, 'rb+') as f:
            pickle.dump(model_list, f)
        self.save(path, model, optimizer, epoch, global_step)

    def load(self, path, model, optimizer, use_cpu=False):
        """
        Return starting epoch and global step
        """

        assert osp.exists(path), 'Checkpoint {} does not exist.'.format(path)
        print('Loading checkpoint from {}...'.format(path))
        if not use_cpu:
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint.pop('model'))
        if optimizer:
            optimizer.load_state_dict(checkpoint.pop('optimizer'))
        print('Checkpoint loaded.')
        return checkpoint

    def load_last(self, path, model, optimizer, use_cpu=False):
        """
        If path is '', we load the last checkpoint
        """

        if path == '':
            with open(self.listfile, 'rb') as f:
                model_list = pickle.load(f)
                if len(model_list) == 0:
                    print('No checkpoint found. Starting from scratch')
                    return None
                else:
                    path = model_list[-1]

        return self.load(path, model, optimizer, use_cpu)

    def save_best(self, metric_name, value, checkpoint,  min_is_better):
        metric_file = os.path.join(self.checkpointdir, 'best_{}.json'.format(metric_name))
        checkpoint_file = os.path.join(self.checkpointdir, 'best_{}.pth'.format(metric_name))

        now = datetime.now()
        log = {
            'name': metric_name,
            'value': float(value),
            'date': now.strftime("%Y-%m-%d %H:%M:%S"),
            'global_step': checkpoint[-1]
        }

        if not os.path.exists(metric_file):
            dump = True
        else:
            with open(metric_file, 'r') as f:
                previous_best = json.load(f)
            if not math.isfinite(log['value']):
                dump = True
            elif (min_is_better and log['value'] < previous_best['value']) or (
                    not min_is_better and log['value'] > previous_best['value']):
                dump = True
            else:
                dump = False
        if dump:
            with open(metric_file, 'w') as f:
                json.dump(log, f)
            self.save(checkpoint_file, *checkpoint)

    def load_best(self, metric_name, model, optimizer, use_cpu=False):
        metric_file = os.path.join(self.checkpointdir, 'best_{}.json'.format(metric_name))
        checkpoint_file = os.path.join(self.checkpointdir, 'best_{}.pth'.format(metric_name))

        assert osp.exists(metric_file), 'Metric file does not exist'
        assert osp.exists(checkpoint_file), 'checkpoint file does not exist'

        return self.load(checkpoint_file, model, optimizer, use_cpu)


class SmoothedValue:
    """
    Record the last several values, and return summaries
    """

    def __init__(self, maxsize=20):
        self.values = deque(maxlen=maxsize)
        self.count = 0
        self.sum = 0.0

    def update(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.values.append(value)
        self.count += 1
        self.sum += value

    @property
    def median(self):
        return np.median(np.array(self.values))

    @property
    def avg(self):
        return np.mean(self.values)

    @property
    def global_avg(self):
        return self.sum / self.count


class MetricLogger:
    def __init__(self):
        self.values = defaultdict(SmoothedValue)

    def update(self, **kargs):
        for key, value in kargs.items():
            self.values[key].update(value)

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, item):
        self.values[key].update(item)


def set_seed(cfg):
    # Seed
    import torch
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(cfg.seed)
