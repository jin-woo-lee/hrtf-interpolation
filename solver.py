#!/usr/bin/env python3
import argparse
import logging
import logging.handlers
import os
import random
import csv
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import datasets
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from utils import *
from torch.autograd import Variable
import torch.optim as optim
import re
from tqdm import tqdm

class Solver(object):
    def __init__(self, args):
        self.total_train_loss = {}
        self.total_valid_loss = {}
        self.mp_context = torch.multiprocessing.get_context('fork')

    def set_gpu(self, args):
        logging.info('set distributed data parallel')
        args.batch_size = int(args.batch_size / args.ngpus_per_node)
        if args.train:
            self.train_sampler = DistributedSampler(self.trainset,shuffle=True,rank=args.gpu,seed=args.seed)
            self.valid_sampler = DistributedSampler(self.validset,shuffle=False,rank=args.gpu)
            self.train_loader = DataLoader(
                self.trainset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                multiprocessing_context=self.mp_context,
                pin_memory=True, sampler=self.train_sampler, drop_last=True,
                worker_init_fn = lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)),
            )
            self.valid_loader = DataLoader(
                self.validset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                multiprocessing_context=self.mp_context,
                pin_memory=True, sampler=self.valid_sampler, drop_last=True,
                worker_init_fn = lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)),
            )
        if args.test:
            self.test_sampler = DistributedSampler(self.testset,shuffle=False,rank=args.gpu)
            self.test_loader = DataLoader(
                self.testset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                multiprocessing_context=self.mp_context,
                pin_memory=True, sampler=self.test_sampler, drop_last=True,
                worker_init_fn = lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)),
            )

        logging.info('set device for model')
        ############################### 
        if args.model=='hyperfilm':
            module = __import__('networks.hyperfilm', fromlist=[''])
            model = module.HyperFiLM(
                in_ch=args.in_ch,
                out_ch=args.out_ch,
                cnn_layers=args.cnn_layers,
                rnn_layers=args.rnn_layers,
                batch_size=args.batch_size,
                condition=args.condition,
                film_dim=args.film_dim,
                without_anm=args.without_anm,
            )
        ############################### 
    
        self.optimizer = torch.optim.AdamW(
            params=list(model.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999),
            amsgrad=False,
            weight_decay=0.01 ,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            self.optimizer,
            lr_lambda=lambda i: args.lr_lambda,
            last_epoch=-1,
            verbose=False,
        )
    
        torch.cuda.set_device(args.gpu)
        # Distribute models to machine
        model = model.to('cuda:{}'.format(args.gpu))
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            output_device=args.gpu,
            #find_unused_parameters=True,
            broadcast_buffers=False,    # required for batchnorm
        )
        self.model = ddp_model

        if args.resume or args.test:
            print("Load checkpoint from: {}".format(args.ckpt))
            checkpoint = torch.load(args.ckpt, map_location=f'cuda:{args.gpu}')
            self.model.load_state_dict(checkpoint["model"])
            self.start_epoch = (int)(checkpoint["epoch"])
            self.step = (int)(checkpoint["step"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            args.k_folds = checkpoint["k_folds"]
            args.test_fold = checkpoint["test_fold"]
        else:
            self.start_epoch = 0
            self.step = 0

    def set_dataset(self, args):
        module = __import__('dataset.loader', fromlist=[''])
        data_dir = args.data_dir
        tot_subj = 93
        test_subj = (tot_subj // args.k_folds) + 1
        subj_list = np.random.permutation(np.arange(tot_subj))
        subj_list = np.roll(subj_list, test_subj*args.test_fold)
        if args.train:
            train_subj = subj_list[test_subj:- test_subj].tolist()
            valid_subj = subj_list[:test_subj].tolist()
            self.trainset = module.Trainset(
                data_dir,
                subj_list=train_subj,
                method='S', ear='L',
                patch_range=args.p_range,
                n_samples=args.in_ch,
                sort_by_dist=False,
            )
            self.validset = module.Testset(
                data_dir,
                subj_list=valid_subj,
                method='S', ear='L',
                patch_range=args.p_range,
                n_samples=args.in_ch,
                mode='Valid',
                sort_by_dist=False,
            )
        if args.test:
            test_subj = subj_list[- test_subj:].tolist()
            self.testset = module.Testset(
                data_dir,
                subj_list=test_subj,
                method='S', ear='L',
                patch_range=args.p_range,
                n_samples=args.in_ch,
                mode='Test',
                sort_by_dist=True,
            )

    def save_checkpoint(self, args, epoch, step, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "step" : step,
            "k_folds": args.k_folds,
            "test_fold": args.test_fold,
        }
        checkpoint_path = os.path.join(checkpoint_dir,'{}_{}.pt'.format(args.model, step))
        torch.save(checkpoint_state, checkpoint_path)
        print("Saved checkpoint: {}".format(checkpoint_path))

    def train(self, args):
    
        self.model.train()
    
        LOSS = []
        logging.info('train start')
        root_dir = f'results/{args.exp_name}/train'
        valid_error = np.inf
        for epoch in range(self.start_epoch, args.total_epochs+1):
            loss_epoch = []
            self.epoch = epoch
            for i, ts in enumerate(self.train_loader):
                self.step = self.epoch*len(self.train_loader) + i
                input, target, an_mes, scoord, tcoord = ts

                inputs = input.cuda(args.gpu,non_blocking=True).float()
                target = target.cuda(args.gpu,non_blocking=True).float()
                an_mes = an_mes.cuda(args.gpu,non_blocking=True).float()
                scoord = scoord.cuda(args.gpu,non_blocking=True).float()
                tcoord = tcoord.cuda(args.gpu,non_blocking=True).float()

                # preprocess
                inputs_m, inputs_p = logmag_phase(inputs)
                target_m, target_p = logmag_phase(target)
                inputs_m = inputs_m / args.rescale
                target_m = target_m / args.rescale

                # model fwd
                estimate, nl = self.model(inputs_m, scoord, tcoord, an_mes)

                # TF loss: magnitude
                loss = F.mse_loss(estimate, target_m).pow(.5)    # extrapolation loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
                loss_epoch.append(loss.item())
                if args.gpu==0:
                    if i % args.plot_iter == 0 and args.plot_train:
                        inp = inputs_m.detach().cpu().numpy() * args.rescale
                        tgt = target_m.detach().cpu().numpy() * args.rescale
                        est = estimate.detach().cpu().numpy() * args.rescale
                        _nl = nl.detach().cpu().numpy() * args.rescale
                        scoord = scoord.detach().cpu().numpy().squeeze(1)
                        tcoord = tcoord.detach().cpu().numpy().squeeze(1)
                        plot_dir = f"{root_dir}/plot/{epoch}/{i}"
                        os.makedirs(plot_dir, exist_ok=True)
                        plots = plot_sample(
                            (inp, tgt, est, _nl),
                            plot_dir,
                            tcoord,
                        )
    
                    if i % args.save_iter == 0:
                        checkpoint_dir = f'{root_dir}/ckpt/{epoch}'
                        self.save_checkpoint(args, epoch, i, checkpoint_dir)
    
                    if i % 1000 == 0:
                        tot_loss = loss.item()
                        self.total_train_loss['total'] = tot_loss
                        _lr = round(self.scheduler.get_last_lr()[0],5)
                        print(
                            f"[Train] epoch #{epoch}/{args.total_epochs},\t"
                            f"step #{i}/{len(self.train_loader)},\t"
                            f"lr {_lr},\t"
                            f"total loss {tot_loss:.3f} | valid error {valid_error:.3f},\t"
                        )
    
            LOSS.append(sum(loss_epoch) / len(self.train_loader))
            # end of each epoch
            if len(LOSS) > 2 and round(LOSS[-2],5) >= round(LOSS[-3],5) and round(LOSS[-1],5) >= round(LOSS[-2],5):
                self.scheduler.step()
            if epoch % args.valid_epoch == 0 and epoch != 0:
                valid_error = self.test(args, 'valid')
                self.model.train()
        # end of training
        checkpoint_dir = f'{root_dir}/ckpt/{epoch}'
        self.save_checkpoint(args, epoch, i, checkpoint_dir)
    
    def test(self, args, mode='test'):

        self.model.eval()
        #logging.info('test start')
        if args.gpu==0:
            print('test start')
        TOT = []
        TDB = []
        FRQ = []
        HOR = []
        MED = []
        FRO = []
        LOG = []
        epoch = self.start_epoch
        root_dir = f'results/{args.exp_name}/{mode}/{epoch}'
        os.makedirs(root_dir, exist_ok=True)
        with torch.no_grad():
            loader = self.valid_loader if mode=='valid' else self.test_loader
            for i, ts in enumerate(tqdm(loader)):
                input, target, an_mes, scoord, tcoord = ts

                inputs = input.cuda(args.gpu,non_blocking=True).float()
                target = target.cuda(args.gpu,non_blocking=True).float()
                an_mes = an_mes.cuda(args.gpu,non_blocking=True).float()
                scoord = scoord.cuda(args.gpu,non_blocking=True).float()
                tcoord = tcoord.cuda(args.gpu,non_blocking=True).float()

                # preprocess
                inputs_m, inputs_p = logmag_phase(inputs)
                target_m, target_p = logmag_phase(target)
                inputs_m = inputs_m / args.rescale
                target_m = target_m / args.rescale

                # model fwd
                estimate, nl = self.model(inputs_m, scoord, tcoord, an_mes)

                # TF loss: magnitude
                loss = F.mse_loss(estimate, target_m).pow(.5)
                tot_loss = loss.item()
                self.total_valid_loss['total'] = tot_loss

                inputs_m *= args.rescale
                estimate *= args.rescale
                target_m *= args.rescale
                loss_db = F.mse_loss(estimate, target_m).pow(.5).item()
                loss_frq = (estimate - target_m).abs().float().squeeze(1).mean(0)
                for b in range(tcoord.shape[0]):
                    x, y, z = tcoord[b].squeeze()
                    ee = (estimate[b] - target_m[b]).pow(2).float().mean().sqrt()
                    if x.abs() < 1e-10: 
                        FRO.append(ee)
                    if y.abs() < 1e-10: 
                        MED.append(ee)
                    if z.abs() < 1e-10: 
                        HOR.append(ee)
                TOT.append(tot_loss)
                TDB.append(loss_db)
                FRQ.append(loss_frq)

                LOG.append('\n'.join([
                    f'[Test] Total RMSE: {loss_db:.4f} dB',
                ]))
    
                if args.gpu==0 and args.plot_test:
                    tf_inputs_m = inputs_m.detach().cpu().numpy()
                    tf_target_m = target_m.detach().cpu().numpy()
                    tf_estimate = estimate.detach().cpu().numpy()
                    _nl = nl.detach().cpu().numpy()*args.rescale
                    scoord = scoord.detach().cpu().numpy().squeeze(1)
                    tcoord = tcoord.detach().cpu().numpy().squeeze(1)
                    plot_dir = f"{root_dir}/plot/{epoch}"
                    os.makedirs(plot_dir, exist_ok=True)
                    plots = plot_sample(
                        (tf_inputs_m, tf_target_m, tf_estimate, _nl),
                        plot_dir,
                        tcoord,
                        test=i,
                    )
    
        total_error = sum(TOT) / len(TOT)
        total_er_db = sum(TDB) / len(TDB)
        freqs_error = sum(FRQ) / len(FRQ)
        horis_error = sum(HOR) / len(HOR)
        medis_error = sum(MED) / len(MED)
        frons_error = sum(FRO) / len(FRO)
        LOG.append('\n'.join([
            f'-'*50,
            f'[Summary] {args.exp_name}: {args.k_folds}_fold-{args.test_fold}-{epoch}',
            f'          Total RMSE: {total_er_db}',
        ]))
        if args.gpu==0:
            print('test finish')
            file_name = f'{args.k_folds}_fold-{args.test_fold}'
            log_path = f'{root_dir}/{file_name}.txt'
            with open(log_path, 'w') as f:
                for ll in LOG:
                    lls = ll.split('\n')
                    for l in lls:
                        f.write(l)
                        f.write('\n')
            freqs_error = freqs_error.detach().cpu().numpy()
            horis_error = horis_error.detach().cpu().numpy()
            medis_error = medis_error.detach().cpu().numpy()
            frons_error = frons_error.detach().cpu().numpy()
            np.save(f'{root_dir}/freqs_error.npy', freqs_error)
            np.save(f'{root_dir}/horis_error.npy', horis_error)
            np.save(f'{root_dir}/medis_error.npy', medis_error)
            np.save(f'{root_dir}/frons_error.npy', frons_error)
            print('saved accuracy log:', log_path)
            print(f'[Summary] {args.exp_name}: {args.k_folds}_fold-{args.test_fold}-{epoch}')
            print(f'          Total RMSE: {total_er_db}')
        return total_error


