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
            model, testset, self.cfg.eval.batch_size,
            self.cfg.eval.num_workers, device, num_samples=None
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
        model.train()

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

        writer.add_scalar('val/count_sparsity', result_dict['count_sparsity'], global_step)
        writer.add_scalar('val/hoyer', result_dict['hoyer'], global_step)
        writer.add_scalar('val/roc_auc', result_dict['roc_auc'], global_step)
        writer.add_scalar('val/inclass_log_like', result_dict['inclass_log_like'], global_step)

        grid_image = make_grid(result_dict['imgs'][:100], 5, normalize=False, pad_value=1)
        writer.add_image('{}/1-image'.format('val'), grid_image, global_step)

        grid_image = make_grid(result_dict['y'][:100], 5, normalize=False, pad_value=1)
        writer.add_image('{}/2-reconstruction'.format('val'), grid_image, global_step)

        writer.add_image('{}/3-z_histogram'.format('val'), result_dict['z_histogram'], global_step)

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
        log_likes = []

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
                log_likes.append(log['log_like'].detach().cpu())

                pbar.update(1)

            # compute sparsity
            count_sparsity = algo_utils.avg_count_sparsity(torch.cat(all_z))
            hoyer = algo_utils.hoyer_metric(torch.cat(all_z))


        result_dict = {
            'hoyer': hoyer,
            'count_sparsity': count_sparsity,
            'imgs': last_imgs,
            'y': last_ys,
            'z': torch.cat(all_z),
            'labels': torch.cat(labels),
            'log_like': torch.cat(log_likes),
        }
        inclass_log_like = result_dict['log_like'][result_dict['labels'] <= 4].mean()
        result_dict['inclass_log_like'] = inclass_log_like.item()

        z = result_dict['z'].numpy()
        log_like = result_dict['log_like'].numpy()
        labels = result_dict['labels'].numpy()

        z_sp = (np.absolute(z) > algo_utils.SPARSE_EPS).astype(np.float)
        z_hist = np.zeros((10, z.shape[-1]))

        in_class = self.cfg.mnist.in_class
        out_class = set(range(self.cfg.mnist.total_num_class)) - \
            set(self.cfg.mnist.in_class)
        for i in in_class:
            z_i = z_sp[labels == i, :]
            pat = np.mean(z_i, axis=0)
            z_hist[i, :] = pat

        for i in out_class:
            z_i = z_sp[labels == i, :]
            pat = np.mean(z_i, axis=0)
            z_hist[i, :] = pat
        z_hist = np.expand_dims(z_hist, axis=0)
        result_dict['z_histogram'] = z_hist

        auc = self.in_out_class_roc(in_class, out_class, log_like, labels)
        result_dict['roc_auc'] = auc
        return result_dict

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
            ('count_sparsity', result_dict['count_sparsity'].item()),
            ('hoyer', result_dict['hoyer'].item()),
            ('roc_auc', result_dict['roc_auc']),
            ('inclass_log_like', result_dict['inclass_log_like']),
        ])
        with open(json_path, 'w') as f:
            json.dump(tosave, f, indent=2)

        print('Results have been saved to {}.'.format(json_path))

    def print_result(self, result_dict, files):
        pass

    def in_out_class_roc(self, in_class, out_class, scores, labels):
        from sklearn.metrics import roc_curve, auc
        def save_fig(fpr, tpr, auc):
            import matplotlib.pyplot as plt
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('In/out ROC')
            plt.legend(loc="lower right")
            plt.savefig('{}.png'.format(self.cfg.exp_name))
        gt = np.zeros_like(labels)
        for i in in_class:
            gt[labels == i] = 1
        fpr, tpr, _ = roc_curve(gt, scores)
        roc_auc = auc(fpr, tpr)
        #save_fig(fpr, tpr, roc_auc)
        return roc_auc
