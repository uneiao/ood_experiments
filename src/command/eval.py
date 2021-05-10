import os
import os.path as osp
import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from model import get_model
from data_loader import get_dataset, get_dataloader
from evaluator import get_evaluator
from solver import get_optimizer
from utils import Checkpointer, MetricLogger
from viz import get_vislogger


def evaluate(cfg):

    print('Experiment name:', cfg.exp_name)
    print('Dataset:', cfg.dataset)
    print('Model name:', cfg.model_name)
    print('Resume:', cfg.resume)
    if cfg.resume:
        print('Checkpoint:', cfg.resume_ckpt if cfg.resume_ckpt else \
                'last checkpoint')
    print('Using device:', cfg.device)
    if 'cuda' in cfg.device:
        print('Using parallel:', cfg.parallel)
    if cfg.parallel:
        print('Device ids:', cfg.device_ids)

    print('Loading data')

    testset = get_dataset(cfg, 'test')
    model = get_model(cfg)
    model = model.to(cfg.device)
    checkpointer = Checkpointer(
        osp.join(cfg.checkpointdir, cfg.exp_name),
        max_num=cfg.train.max_ckpt)
    evaluator = get_evaluator(cfg)
    model.eval()

    use_cpu = 'cpu' in cfg.device
    if cfg.resume_ckpt:
        checkpoint = checkpointer.load(
            cfg.resume_ckpt, model, None, None, use_cpu)
    elif cfg.eval.checkpoint == 'last':
        checkpoint = checkpointer.load_last('', model, None, use_cpu)
    elif cfg.eval.checkpoint == 'best':
        checkpoint = checkpointer.load_best(
            cfg.eval.metric, model, None, None, use_cpu)

    if cfg.parallel:
        model = nn.DataParallel(model, device_ids=cfg.device_ids)


    evaldir = osp.join(cfg.evaldir, cfg.exp_name)
    info = {
        'exp_name': cfg.exp_name
    }
    evaluator.test_eval(
        model, testset, cfg.device, evaldir, info)
