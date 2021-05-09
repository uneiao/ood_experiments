# -*- coding:utf8 -*-
#! /usr/bin/python3

import argparse

import command
import config
import utils


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        metavar='MODE',
        help='train/eval commands'
    )

    parser.add_argument(
        '--config_file',
        type=str,
        default='',
        metavar='FILE_PATH',
        help='config file path'
    )

    parser.add_argument(
        'opts',
        help='other config options',
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()
    cfg = config.init_config()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)

    return cfg, args.mode


def main():
    commands = {
        'train': command.train,
        'eval': command.test,
    }
    cfg, task = parse_args()
    utils.set_seed(cfg)
    commands[task](cfg)


if __name__ == '__main__':
    main()
