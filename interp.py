import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm
from utils import *

def load_state(args, model):
    print("...\t Load checkpoint from: {}".format(args.ckpt))
    checkpoint = torch.load(args.ckpt, map_location=f'cuda:{args.gpu}')
    filtered_dicts = {}
    for k in checkpoint['model'].keys():
        if 'module.' in k:
            filtered_dicts[k.replace('module.','')] = checkpoint['model'][k]
    model.load_state_dict(filtered_dicts)
    args.k_folds = checkpoint["k_folds"]
    args.test_fold = checkpoint["test_fold"]
    print(f"...\t {args.test_fold}-th fold of {args.k_folds}-fold cross-validation")

def interp(args):
    print("="*30)
    print("    Interpolation Test")
    print("="*30)
    
    #============================== 
    # Load Model
    #============================== 
    print("... Load model")
    module = __import__('networks.hyperfilm', fromlist=[''])
    model = module.HyperFiLM(
        in_ch=args.in_ch,
        out_ch=args.out_ch,
        cnn_layers=args.cnn_layers,
        rnn_layers=args.rnn_layers,
        batch_size=args.batch_size,
        condition=args.condition,
    )
    model = model.cuda(args.gpu)
    load_state(args, model)
    
    file_name = f'{args.k_folds}_fold-{args.test_fold}'
    save_dir = f'results/{args.exp_name}/test/interp/{args.epoch}'
    plot_dir = f'results/{args.exp_name}/test/interp/{args.epoch}/plot'
    log_path = f'{save_dir}/{file_name}.txt'
    os.makedirs(plot_dir, exist_ok=True)
    
    #============================== 
    # Load Data
    #============================== 
    print("... Load data")
    module = __import__('dataset.loader', fromlist=[''])
    data_dir = '/data2/HUTUBS/HRIRs-pos'
    tot_subj = 93
    test_subj = (tot_subj // args.k_folds) + 1
    subj_list = np.random.permutation(np.arange(tot_subj))
    subj_list = np.roll(subj_list, test_subj*args.test_fold)
    test_subj = subj_list[- test_subj:].tolist()
    testset = module.Testset(
        data_dir,
        subj_list=test_subj,
        method='S', ear='L',
        patch_range=args.p_range,
        n_samples=args.in_ch,
        mode='Test',
        sort_by_dist=True,
    )
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    print("... Test start")
    model.eval()
    LOSS = []
    for i, ts in enumerate(tqdm(test_loader)):
        input, target, an_mes, scoord, tcoord = ts
        inputs = input.cuda(args.gpu,non_blocking=True).float()
        target = target.cuda(args.gpu,non_blocking=True).float()
        an_mes = an_mes.cuda(args.gpu,non_blocking=True).float()
        scoord = scoord.cuda(args.gpu,non_blocking=True).float()
        tcoord = tcoord.cuda(args.gpu,non_blocking=True).float()
    
        inp_m, inputs_p = logmag_phase(inputs)
        tar_m, target_p = logmag_phase(target)
        inp_m = inp_m / args.rescale
        tar_m = tar_m / args.rescale
    
        delta = (tcoord - scoord[:,:,:3]) / 3
        tcoord_0 = tcoord - 0*delta    # == tcoord
        tcoord_1 = tcoord - 1*delta    # ~= tcoord
        tcoord_2 = tcoord - 2*delta    # ~= scoord
        tcoord_3 = tcoord - 3*delta    # == scoord
        with torch.no_grad():
            est_0, _ = model(inp_m, scoord, tcoord_0, an_mes)
            est_1, _ = model(inp_m, scoord, tcoord_1, an_mes)
            est_2, _ = model(inp_m, scoord, tcoord_2, an_mes)
            est_3, _ = model(inp_m, scoord, tcoord_3, an_mes)

        loss = F.mse_loss(args.rescale * est_0, args.rescale * tar_m).pow(.5)
        _loss = round(loss.item(), 5)
        LOSS.append(_loss)

        B = est_0.shape[0]
        out_np_0 = est_0.detach().cpu().numpy() * args.rescale
        out_np_1 = est_1.detach().cpu().numpy() * args.rescale
        out_np_2 = est_2.detach().cpu().numpy() * args.rescale
        out_np_3 = est_3.detach().cpu().numpy() * args.rescale
        inpt_np = inp_m.detach().cpu().numpy() * args.rescale
        targ_np = tar_m.detach().cpu().numpy() * args.rescale
        freqs = np.linspace(0,22050,targ_np.shape[-1])
        for b in range(B):
            figure = plt.figure(figsize=(13,7))
            #---------- 
            plt.subplot(211)
            in_ch = inpt_np.shape[1]
            for j in range(in_ch):
                plt.plot(freqs, inpt_np[b,j], label=f'input-{j}', color='k', linewidth=.3)
            plt.plot(freqs, targ_np[b,0], label='target', color='b', linewidth=.8)
            plt.plot(freqs, out_np_0[b,0], label='estimate', color='r', linewidth=.8)
            plt.legend()
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude (dB)")
            plt.axhline(y=0, color='k', linewidth=.5)
            plt.tight_layout()
            #---------- 
            plt.subplot(212)
            plt.plot(freqs, inpt_np[b,0],  color='r', label='input', linewidth=.8)
            plt.plot(freqs, out_np_3[b,0], color='y', label='interp-3', linewidth=.8)
            plt.plot(freqs, out_np_2[b,0], color='g', label='interp-2', linewidth=.8)
            plt.plot(freqs, out_np_1[b,0], color='c', label='interp-1', linewidth=.8)
            plt.plot(freqs, out_np_0[b,0], color='b', label='interp-0', linewidth=.8)
            plt.plot(freqs, targ_np[b,0],  color='k', label='target', linewidth=.8)
            plt.legend()
            plt.axhline(y=0, color='k', linewidth=.5)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude (dB)")
            plt.tight_layout()
            #---------- 
            plt.savefig(f'{plot_dir}/{i}-{b}.png')
            plt.close()
    with open(log_path,'a') as log:
        log.write(f"[Test] loss: {sum(LOSS)/len(testset)}")
        log.write("\n")
    print(f"[Test] loss: {sum(LOSS)/len(testset)}")

if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--epoch', default=0, type=int)
    parser.add_argument('--step', default=0, type=int)
    parser.add_argument('--model', default='hyperfilm', type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--start_iter', default=0, type=int, help="initial number of iteration")
    parser.add_argument('--in_ch', type=int, default=16)
    parser.add_argument('--out_ch', type=int, default=1)
    parser.add_argument('--cnn_layers', type=int, default=5)
    parser.add_argument('--rnn_layers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of inference')
    parser.add_argument('--k_folds', type=int, default=10, help='number of folds for cross-validation')
    parser.add_argument('--test_fold', type=int, default=10, help='k for test')
    parser.add_argument('--p_range', default=0.3, type=float, help='patch range')
    parser.add_argument('--rescale', default=50, type=float, help='rescale factor for input')
    parser.add_argument('--condition', type=str, default='hyper')
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    args = parser.parse_args()

    args.ckpt = f'results/{args.exp_name}/train/ckpt/{args.epoch}/{args.model}_{args.step}.pt'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    interp(args)
 
