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
from torchvision.utils import make_grid

from utils import MetricLogger
import algo_utils


def get_evaluator(cfg):
    return PerfEvaluator(cfg)


class PerfEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg

    @torch.no_grad()
    def test_eval(self, model, testset, device, evaldir, info):
        result_dict = self.eval_targets(
            model, testset, self.cfg.test.batch_size,
            self.cfg.test.num_workers, device, num_samples=None
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
            model, valset, self.cfg.train.batch_size, self.cfg.train.num_workers,
            device, num_samples=None, global_step=global_step,
        )
        count_sparsity = result_dict['count_sparsity']
        hoyer = result_dict['hoyer']
        z = result_dict['z'].numpy()
        labels = result_dict['labels'].numpy()

        z_sp = (np.absolute(z) > algo_utils.EPS).astype(np.float)
        z_hist = np.zeros((10, z.shape[-1]))
        for i in self.cfg.mnist.filtering_class:
            z_i = z_sp[labels == i, :]
            pat = np.mean(z_i, axis=0)
            z_hist[i, :] = pat

        for i in set(range(10)) - set(self.cfg.mnist.filtering_class):
            z_i = z_sp[labels == i, :]
            pat = np.mean(z_i, axis=0)
            z_hist[i, :] = pat
        z_hist = np.expand_dims(z_hist, axis=0)

        writer.add_scalar('val/count_sparsity', count_sparsity, global_step)
        writer.add_scalar('val/hoyer', hoyer, global_step)

        grid_image = make_grid(result_dict['imgs'][:100], 5, normalize=False, pad_value=1)
        writer.add_image('{}/1-image'.format('val'), grid_image, global_step)

        grid_image = make_grid(result_dict['y'][:100], 5, normalize=False, pad_value=1)
        writer.add_image('{}/2-reconstruction'.format('val'), grid_image, global_step)

        writer.add_image('{}/3-z_histogram'.format('val'), z_hist, global_step)

        return result_dict

    def eval_targets(
            self,
            model,
            dataset,
            batch_size,
            num_workers,
            device,
            num_samples=None,
            global_step=0,
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

        last_imgs = None
        last_ys = None
        all_z = []
        labels = []
        model.eval()
        with torch.no_grad():
            pbar = tqdm(total=len(dataloader))
            for i, (imgs, lbs) in enumerate(dataloader):
                last_imgs = imgs.detach().cpu()
                imgs = imgs.to(device)

                loss, log = \
                    model(imgs, global_step=global_step or 100000000)

                last_ys = log['y'].detach().cpu()
                all_z.extend(log['z'].unsqueeze(dim=0).detach().cpu())
                labels.append(lbs.detach().cpu())

                pbar.update(1)

            # compute sparsity
            count_sparsity = algo_utils.avg_count_sparsity(torch.cat(all_z))
            hoyer = algo_utils.hoyer_metric(torch.cat(all_z))

        model.train()

        return {
            'hoyer': hoyer,
            'count_sparsity': count_sparsity,
            'imgs': last_imgs,
            'y': last_ys,
            'z': torch.cat(all_z),
            'labels': torch.cat(labels),
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
            ('count_sparsity', list(result_dict['count_sparsity'])),
            ('hoyer', list(result_dict['hoyer'])),
        ])
        with open(json_path, 'w') as f:
            json.dump(tosave, f, indent=2)

        print('Results have been saved to {}.'.format(json_path))

    def print_result(self, result_dict, files):
        pass
