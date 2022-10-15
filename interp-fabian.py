import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_dict
import sofa
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

def interp(args, return_phs=False):
    print("="*30)
    print("    Interpolation Test: FABIAN")
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
        film_dim=args.film_dim,
    )
    model = model.cuda(args.gpu)
    load_state(args, model)
    
    if args.save_data:
        file_name = f'{args.k_folds}_fold-{args.test_fold}'
        save_dir = f'results/{args.exp_name}/test/fabian-interp/{args.epoch}'
        plot_dir = f'results/{args.exp_name}/test/fabian-interp/{args.epoch}/plot'
        errr_dir = f'results/{args.exp_name}/test/fabian-interp/{args.epoch}/error'
        log_path = f'{save_dir}/{file_name}.txt'
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(errr_dir, exist_ok=True)
    
    #============================== 
    # Load Data
    #============================== 
    print("... Load data")
    module = __import__('dataset.loader', fromlist=[''])
    ############################## 
    if args.x_constraint is not None:
        cons = 'fro'
    elif args.y_constraint is not None:
        cons = 'med'
    elif args.z_constraint is not None:
        cons = 'hor'
    else:
        cons = 'full'
    scale_factor=args.scale_factor
    name = f'FAB_C2F_{cons}_{scale_factor}'
    src_subj = args.source if args.source else '/data2/HRTF/FABIAN/FABIAN_HRTF_DATABASE_V1/1 HRIRs/SOFA/FABIAN_HRIR_modeled_HATO_0.sofa'
    an_mes = args.measures if args.measures else load_dict('/data2/HRTF/HUTUBS/pkl-15/1.pkl')['label']['L']['feature']
    num_grid = 11950
    ############################## 
    #name = 'H2F-sim'
    #src_subj = '/data2/HUTUBS/HRIRs/pp1_HRIRs_simulated.sofa'
    #tar_subj = '/data2/FABIAN_HRTF_DATABASE_V1/1 HRIRs/SOFA/FABIAN_HRIR_modeled_HATO_0.sofa'
    #an_mes = load_dict('/data2/HUTUBS/HRIRs-pos/1.pkl')['label']['L']['feature']
    #src_grid = 1730
    #tar_grid = 11950
    #scale_factor=1
    ############################## 
    #name = 'H2F-mes'
    #src_subj = '/data2/HUTUBS/HRIRs/pp1_HRIRs_measured.sofa'
    #tar_subj = '/data2/FABIAN_HRTF_DATABASE_V1/1 HRIRs/SOFA/FABIAN_HRIR_measured_HATO_0.sofa'
    #an_mes = load_dict('/data2/HUTUBS/HRIRs-pos/1.pkl')['label']['L']['feature']
    #src_grid = 440
    #tar_grid = 11950
    ############################## 
    testset = module.CoarseSofaSet(
        name=name,
        sofa_path=src_subj,
        anthropometry=an_mes,
        patch_range=args.p_range,
        n_samples=args.in_ch,
        num_grid = num_grid,
        sort_by_dist=True,
        x_constraint=args.x_constraint,
        y_constraint=args.y_constraint,
        z_constraint=args.z_constraint,
        scale_factor=scale_factor,
    )
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    print("... Test start")
    sd_EST = [];   sd_LIN = []
    rmse_EST = []; rmse_LIN = []
    EST = []; TAR = []; LIN = []; PHS = []
    model.eval()
    for i, ts in enumerate(tqdm(test_loader)):
        input, target, linear, an_mes, scoord, tcoord = ts
        inputs = input.cuda(args.gpu,non_blocking=True).float()     # (batch, in_ch, 256)
        target = target.cuda(args.gpu,non_blocking=True).float()    # (batch, 1, 256)
        linear = linear.cuda(args.gpu,non_blocking=True).float()    # (batch, 1, 256)
        an_mes = an_mes.cuda(args.gpu,non_blocking=True).float()    # (batch, 1, 12)
        scoord = scoord.cuda(args.gpu,non_blocking=True).float()    # (batch, 1, 3*in_ch)
        tcoord = tcoord.cuda(args.gpu,non_blocking=True).float()    # (batch, 1, 3)
    
        inputs_m, inputs_p = logmag_phase(inputs)
        target_m, target_p = logmag_phase(target)
        linear_m, linear_p = logmag_phase(linear)
        inputs_m = inputs_m / args.rescale
        target_m = target_m / args.rescale
        linear_m = linear_m / args.rescale
    
        with torch.no_grad():
            estimate, nl = model(inputs_m, scoord, tcoord, an_mes)
        estimate *= args.rescale
        target_m *= args.rescale
        linear_m *= args.rescale
    
        rmse_est = F.mse_loss(estimate, target_m).pow(.5)
        rmse_lin = F.mse_loss(linear_m, target_m).pow(.5)
        sd_est = F.l1_loss(estimate, target_m)
        sd_lin = F.l1_loss(linear_m, target_m)
        rmse_EST.append(rmse_est.item())
        rmse_LIN.append(rmse_lin.item())
        sd_EST.append(sd_est.item())
        sd_LIN.append(sd_lin.item())
        for b in range(estimate.shape[0]):
            EST.append(estimate[b])
            TAR.append(target_m[b])
            LIN.append(linear_m[b])
            PHS.append(target_p[b])
    S = 1
    G = testset.tar_sel_grid
    R = scale_factor
    EST = torch.cat(EST, dim=0).reshape(S,G,129)
    TAR = torch.cat(TAR, dim=0).reshape(S,G,129)
    LIN = torch.cat(LIN, dim=0).reshape(S,G,129)
    PHS = torch.cat(PHS, dim=0).reshape(S,G,129,2)

    EST_np = EST.detach().cpu().numpy()
    TAR_np = TAR.detach().cpu().numpy()
    LIN_np = LIN.detach().cpu().numpy()
    PHS_np = PHS.detach().cpu().numpy()

    sd_fr_EST = (EST - TAR).abs().mean(0).mean(-1).detach().cpu().numpy()
    sd_fr_LIN = (LIN - TAR).abs().mean(0).mean(-1).detach().cpu().numpy()
    rmse_fr_EST = (EST - TAR).pow(2).mean(0).mean(-1).pow(.5).detach().cpu().numpy()
    rmse_fr_LIN = (LIN - TAR).pow(2).mean(0).mean(-1).pow(.5).detach().cpu().numpy()
    x = np.linspace(0,360,G)
    y = np.linspace(0,22050,129)
    if args.save_data:
        os.makedirs(f'{plot_dir}/{name}', exist_ok=True)
        deg_lab = 'Azimuth' if args.z_constraint is not None else 'Elevation'
        for s in range(S):
            plt.figure(figsize=(7,13))
            librosa.display.specshow(EST_np[s].T, x_coords=x, y_coords=y, cmap='bwr')
            plt.xticks(np.arange(0,361,60))
            plt.yticks(np.arange(0,22050,4000))
            plt.xlabel(f'{deg_lab} (deg)')
            plt.ylabel('Frequency (Hz)')
            plt.clim([-20,20])
            plt.colorbar()
            plt.savefig(f'{plot_dir}/{name}/{s}-est.png')
            plt.close()

            plt.figure(figsize=(7,13))
            librosa.display.specshow(LIN_np[s].T, x_coords=x, y_coords=y, cmap='bwr')
            plt.xticks(np.arange(0,361,60))
            plt.yticks(np.arange(0,22050,4000))
            plt.xlabel(f'{deg_lab} (deg)')
            plt.ylabel('Frequency (Hz)')
            plt.clim([-20,20])
            plt.colorbar()
            plt.savefig(f'{plot_dir}/{name}/{s}-lin.png')
            plt.close()

            plt.figure(figsize=(7,13))
            librosa.display.specshow(TAR_np[s].T, x_coords=x, y_coords=y, cmap='bwr')
            plt.xticks(np.arange(0,361,60))
            plt.yticks(np.arange(0,22050,4000))
            plt.xlabel(f'{deg_lab} (deg)')
            plt.ylabel('Frequency (Hz)')
            plt.clim([-20,20])
            plt.colorbar()
            plt.savefig(f'{plot_dir}/{name}/{s}-tar.png')
            plt.close()

    if args.save_data:
        with open(log_path,'a') as log:
            log.write(f"[Test] Model RMSE  : {sum(rmse_EST)/len(rmse_EST)} dB\n")
            log.write(f"       Linear RMSE : {sum(rmse_LIN)/len(rmse_LIN)} dB\n")
            log.write(f"       Model SD    : {sum(sd_EST)/len(sd_EST)} dB\n")
            log.write(f"       Linear SD   : {sum(sd_LIN)/len(sd_LIN)} dB\n")
    print(f"[Test] Model RMSE  : {sum(rmse_EST)/len(rmse_EST)} dB")
    print(f"       Linear RMSE : {sum(rmse_LIN)/len(rmse_LIN)} dB")
    print(f"       Model SD    : {sum(sd_EST)/len(sd_EST)} dB")
    print(f"       Linear SD   : {sum(sd_LIN)/len(sd_LIN)} dB")
    if args.save_data:
        np.save(f'{errr_dir}/{name}-est.npy', EST_np)
        np.save(f'{errr_dir}/{name}-lin.npy', LIN_np)
        np.save(f'{errr_dir}/{name}-tar.npy', TAR_np)
        np.save(f'{errr_dir}/{name}-est-sd.npy', sd_fr_EST)
        np.save(f'{errr_dir}/{name}-lin-sd.npy', sd_fr_LIN)
        np.save(f'{errr_dir}/{name}-est-rmse.npy', rmse_fr_EST)
        np.save(f'{errr_dir}/{name}-lin-rmse.npy', rmse_fr_LIN)

    if return_phs:
        return TAR_np[0], EST_np[0], PHS_np[0]
    else:
        return TAR_np[0], EST_np[0]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--step', default=1485, type=int)
    parser.add_argument('--model', default='hyperfilm', type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--start_iter', default=0, type=int, help="initial number of iteration")
    parser.add_argument('--in_ch', type=int, default=16)
    parser.add_argument('--out_ch', type=int, default=1)
    parser.add_argument('--cnn_layers', type=int, default=5)
    parser.add_argument('--rnn_layers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of inference')
    parser.add_argument('--k_folds', type=int, default=5, help='number of folds for cross-validation')
    parser.add_argument('--test_fold', type=int, default=5, help='k for test')
    parser.add_argument('--p_range', default=0.6, type=float, help='patch range')
    parser.add_argument('--rescale', default=50, type=float, help='rescale factor for input')
    parser.add_argument('--condition', type=str, default='hyper')
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--x_constraint', default=None, type=float)
    parser.add_argument('--y_constraint', default=None, type=float)
    parser.add_argument('--z_constraint', default=None, type=float)
    parser.add_argument('--film_dim', type=str, default='chan')
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--save_data', action="store_true")
    parser.add_argument('--show_plot', action="store_true")
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--measures', nargs='+', default=None)
    args = parser.parse_args()

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

    if args.ckpt is None:
        args.ckpt = f'results/{args.exp_name}/train/ckpt/{args.epoch}/{args.model}_{args.step}.pt'
    if args.measures:
        args.measures = [float(m) for m in args.measures[0].split(',')]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    interp(args)
 
