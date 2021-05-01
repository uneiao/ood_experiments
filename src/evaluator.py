from datetime import datetime
import json
import math
import os, sys
import os.path as osp

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import MetricLogger


class PerfEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg

    @torch.no_grad()
    def test_eval(self, model, testset, device, evaldir, info):
        result_dict = self.eval_targets(
            model, testset, eval_cfg.test.batch_size,
            eval_cfg.test.num_workers, device, num_samples=None
        )
        os.makedirs(evaldir, exist_ok=True)
        path = osp.join(
            evaldir,
            'results_{}.json'.format(
                datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
        self.save_to_json(result_dict, path, info)
        #self.print_result(result_dict, [sys.stdout, open('./results.txt', 'w')])

    @torch.no_grad()
    def train_eval(
        self, model, valset, writer, global_step, device,
        checkpoint, checkpointer):
        results = self.train_eval_impl(model, valset, writer, global_step, device)
        #checkpointer.save_best(
        #    'sparseness', results['sparseness'], checkpoint, min_is_better=False)

    @torch.no_grad()
    def train_eval_impl(
        self, model, valset, writer, global_step, device):
        """
        Evaluate any indices during training

        :return: result_dict
        """
        result_dict = self.eval_targets(
            model, valset, eval_cfg.train.batch_size, eval_cfg.train.num_workers,
            device, num_samples=None
        )
        sparseness = result_dict['sparseness']

        writer.add_scalar('val/sparseness', sparseness, global_step)

        return result_dict

    def eval_targets(
            self,
            model,
            dataset,
            batch_size,
            num_workers,
            device,
            num_samples=None,
    ):
        """
        Evaluate target metrics.
        """

        from tqdm import tqdm
        import sys

        model.eval()

        if num_samples is None:
            num_samples = len(dataset)
        dataset = Subset(dataset, indices=range(num_samples))
        dataloader = DataLoader(
            dataset, batch_size=batch_size,
            num_workers=num_workers, shuffle=False)

        model.eval()
        with torch.no_grad():
            pbar = tqdm(total=len(dataloader))
            for i, imgs in enumerate(dataloader):
                imgs = imgs.to(device)

                loss, log = \
                    model(imgs, global_step=100000000)

                # compute sparseness
                avg_count_sparseness = get_avg_count_sparseness(log['z'])

                pbar.update(1)

        model.train()

        return {
            'sparseness': avg_count_sparseness,
        }

    def save_to_json(self, result_dict, json_path, info):
        """
        Save evaluation results to json file

        :param result_dict: a dictionary
        :param json_path: checkpointdir
        :param info: any other thing you want to save
        :return:
        """
        from collections import OrderedDict
        import json
        from datetime import datetime
        tosave = OrderedDict([
            ('date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ('info', info),
            ('sparseness', list(result_dict['sparseness'])),
        ])
        with open(json_path, 'w') as f:
            json.dump(tosave, f, indent=2)

        print('Results have been saved to {}.'.format(json_path))

    def print_result(self, result_dict, files):
        pass


def get_avg_count_sparseness(z, eps=1e-5):
    return (z.abs() < eps).sum() / torch.ones(z.shape).sum()
