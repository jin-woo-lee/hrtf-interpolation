#!/usr/bin/env python3
import argparse
import os
from train import train
from test import test
import logging
import logging.handlers
import torch
import numpy as np

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #------------------------------ 
    # General
    #------------------------------ 
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--plot_iter', type=int, default=500)
    parser.add_argument('--plot_train', action="store_true", help='plot train samples')
    parser.add_argument('--plot_test',  action="store_true", help='plot test samples')
    parser.add_argument('--board_iter', type=int, default=500)
    parser.add_argument('--save_iter', type=int, default=2000)
    parser.add_argument('--load_epoch', type=int, default=0)
    parser.add_argument('--load_step', type=int, default=0)
    parser.add_argument('--total_epochs', type=int, default=100)
    parser.add_argument('--valid_epoch', type=int, default=1)
    parser.add_argument('--ckpt', type=str, default=None, help='load checkpoint')
    parser.add_argument('--test_dir', type=str, default=None, help='directory path of ImageNet dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of inference')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_lambda', type=float, default=0.5)
    parser.add_argument('--k_folds', type=int, default=5, help='number of folds for cross-validation')
    parser.add_argument('--test_fold', type=int, default=5, help='k for test')
    parser.add_argument('--model', type=str, default='hyperfilm')
    parser.add_argument('--exp_name', type=str, default=None)
    #------------------------------ 
    # DDP
    #------------------------------ 
    parser.add_argument('--gpus', nargs='+', default=[0,1], help='gpus')
    parser.add_argument('--n_nodes', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--port', default='1234', type=str, help='port')
    #------------------------------ 
    # Network
    #------------------------------ 
    parser.add_argument('--cnn_layers', type=int, default=5)
    parser.add_argument('--rnn_layers', type=int, default=0)
    parser.add_argument('--in_ch', type=int, default=8)
    parser.add_argument('--out_ch', type=int, default=1)
    parser.add_argument('--condition', type=str, default='hyper')
    parser.add_argument('--film_dim', type=str, default='chan')
    parser.add_argument('--without_anm', action='store_true')
    #------------------------------ 
    # Data
    #------------------------------ 
    parser.add_argument('--p_range', default=0.3, type=float, help='patch range')
    parser.add_argument('--rescale', default=50, type=float, help='rescale factor for input')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--data_dir', default='/ssd/data/HUTUBS/pkl-15', type=str)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_num) for gpu_num in args.gpus])
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = args.port

    logger = logging.getLogger()
    logger.setLevel("DEBUG")
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel("INFO")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if args.exp_name is None:
        args.exp_name = '-'.join([
            f'{args.model}',
            f'{args.cnn_layers}x{args.rnn_layers}',
            f'Fi_{args.film_dim}',
            f'Co_{args.condition}',
            f'd_{args.p_range}',
            f'N_{args.in_ch}',
            f'{args.k_folds}fold{args.test_fold}',
        ])

    if args.train:
        if args.resume and args.ckpt==None:
            ckpt_path = f'results/{args.exp_name}/train/ckpt/{args.load_epoch}/{args.model}_{args.load_step}.pt'
            if os.path.exists(ckpt_path):
                args.ckpt = ckpt_path
            else:
                raise FileNotFoundError(
                    "Specify checkpoint by '--ckpt=...'.",
                    "Otherwise provide exact setting for exp_name, load_epoch, load_step.",
                    ckpt_path
                )
        train(args)
    if args.test:
        if args.ckpt==None:
            ckpt_path = f'results/{args.exp_name}/train/ckpt/{args.load_epoch}/{args.model}_{args.load_step}.pt'
            if os.path.exists(ckpt_path):
                args.ckpt = ckpt_path
            else:
                raise FileNotFoundError(
                    f"Specify checkpoint by '--ckpt=...'. "
                    f"Otherwise provide exact setting for exp_name, load_epoch, load_step. {cpkt_path}"
                )
        test(args)


