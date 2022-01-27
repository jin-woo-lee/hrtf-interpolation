import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_dict

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

def recon(args):
    print("="*30)
    print("    Reconstruction Test: FABIAN")
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
    
    file_name = f'{args.k_folds}_fold-{args.test_fold}'
    save_dir = f'results/{args.exp_name}/test/fabian/{args.epoch}'
    plot_dir = f'results/{args.exp_name}/test/fabian/{args.epoch}/plot'
    log_path = f'{save_dir}/{file_name}.txt'
    os.makedirs(plot_dir, exist_ok=True)
    
    #============================== 
    # Load Data
    #============================== 
    print("... Load data")
    module = __import__('dataset.loader', fromlist=[''])
    #test_subj = '/data2/FABIAN_HRTF_DATABASE_V1/1 HRIRs/SOFA/FABIAN_HRIR_measured_HATO_0.sofa'
    test_subj = '/data2/FABIAN_HRTF_DATABASE_V1/1 HRIRs/SOFA/FABIAN_HRIR_modeled_HATO_0.sofa'
    an_mes = load_dict('/data2/HUTUBS/HRIRs-pos/1.pkl')['label']['L']['feature']
    testset = module.SofaSet(
        name='fabian',
        subj_path=test_subj,
        anthropometry=an_mes,
        patch_range=args.p_range,
        n_samples=args.in_ch,
        num_grid = 11950,
        sort_by_dist=True,
        x_constraint=args.x_constraint,
        y_constraint=args.y_constraint,
        z_constraint=args.z_constraint,
    )
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    print("... Test start")
    model.eval()
    RMSE = []
    SD = []
    for i, ts in enumerate(tqdm(test_loader)):
        input, target, an_mes, scoord, tcoord = ts
        inputs = input.cuda(args.gpu,non_blocking=True).float()     # (batch, sel_grid, in_ch, 256)
        target = target.cuda(args.gpu,non_blocking=True).float()    # (batch, sel_grid, 1, 256)
        an_mes = an_mes.cuda(args.gpu,non_blocking=True).float()    # (batch, sel_grid, 1, 12)
        scoord = scoord.cuda(args.gpu,non_blocking=True).float()    # (batch, sel_grid, 1, 3*in_ch)
        tcoord = tcoord.cuda(args.gpu,non_blocking=True).float()    # (batch, sel_grid, 1, 3)
    
        inputs = inputs.permute(1,0,2,3)    # (sel_grid, batch, in_ch, 256)
        target = target.permute(1,0,2,3)    # (sel_grid, batch, 1, 256)
        an_mes = an_mes.permute(1,0,2,3)    # (sel_grid, batch, 1, 12)
        scoord = scoord.permute(1,0,2,3)    # (sel_grid, batch, 1, 3*in_ch)
        tcoord = tcoord.permute(1,0,2,3)    # (sel_grid, batch, 1, 3)
    
        sel_grid = inputs.shape[0]
        #INP = []
        TAR = []
        EST = []
        est_ERR = []    # model estimation error
        int_ERR = []    # linear interpolation error
        for az in range(sel_grid):
            inputs_m, inputs_p = logmag_phase(inputs[az])
            target_m, target_p = logmag_phase(target[az])
            inputs_m = inputs_m / args.rescale
            target_m = target_m / args.rescale
    
            with torch.no_grad():
                estimate, nl = model(inputs_m, scoord[az], tcoord[az], an_mes[az])
            rmse = F.mse_loss(args.rescale * estimate, args.rescale * target_m).pow(.5)
            sd = F.l1_loss(args.rescale * estimate, args.rescale * target_m)
            _rmse = round(rmse.item(), 5)
            _sd = round(sd.item(), 5)
            RMSE.append(_rmse)
            SD.append(_sd)
            #INP.append(inputs_m.unsqueeze(0))
            TAR.append(target_m.unsqueeze(0))
            EST.append(estimate.unsqueeze(0))

        if (args.x_constraint is not None) or (args.y_constraint is not None) or (args.z_constraint is not None):
            #inp = torch.cat(INP, dim=0).permute(1,0,2,3)    # (batch, sel_grid, in_ch, 129)
            tar = torch.cat(TAR, dim=0).squeeze(2).permute(1,0,2)    # (batch, sel_grid, 129)
            est = torch.cat(EST, dim=0).squeeze(2).permute(1,0,2)    # (batch, sel_grid, 129)
            interp = torch.zeros_like(tar).permute(0,2,1)
            interp[:,:,0::2] = F.interpolate(tar.permute(0,2,1)[:,:,0::2],size=sel_grid,mode='linear')[:,:,0::2]
            interp[:,:,1::2] = F.interpolate(tar.permute(0,2,1)[:,:,1::2],size=sel_grid,mode='linear')[:,:,1::2]
            interp = interp.permute(0,2,1)    # (batch, sel_grid, 129)
            B = est.shape[0]
            #inputs_np = inp.detach().cpu().numpy() * args.rescale
            target_np = tar.detach().cpu().numpy() * args.rescale
            output_np = est.detach().cpu().numpy() * args.rescale
            interp_np = interp.detach().cpu().numpy() * args.rescale
            x = np.linspace(0,360,sel_grid)
            y = np.linspace(0,22050,129)
            if args.z_constraint is not None:
                degs = 'Azimuth'
            elif args.y_constraint is not None:
                degs = 'Elevation'
            for b in range(B):
                plt.figure(figsize=(16,7))
                plt.subplot(131)
                plt.title(f'Output')
                librosa.display.specshow(output_np[b].T, x_coords=x, y_coords=y, cmap='bwr')
                plt.xticks(np.arange(0,361,60))
                plt.yticks(np.arange(0,22050,4000))
                plt.xlabel(f'{degs} (deg)')
                plt.ylabel('Frequency (Hz)')
                #plt.yscale('log')
                plt.clim([-20,20])
                plt.colorbar()

                plt.subplot(132)
                plt.title(f'Linear')
                librosa.display.specshow(interp_np[b].T, x_coords=x, y_coords=y, cmap='bwr')
                plt.xticks(np.arange(0,361,60))
                plt.yticks(np.arange(0,22050,4000))
                plt.xlabel(f'{degs} (deg)')
                plt.ylabel('Frequency (Hz)')
                #plt.yscale('log')
                plt.clim([-20,20])
                plt.colorbar()
            
                plt.subplot(133)
                plt.title(f'Target')
                librosa.display.specshow(target_np[b].T, x_coords=x, y_coords=y, cmap='bwr')
                plt.xticks(np.arange(0,361,60))
                plt.yticks(np.arange(0,22050,4000))
                plt.xlabel(f'{degs} (deg)')
                plt.ylabel('Frequency (Hz)')
                #plt.yscale('log')
                plt.clim([-20,20])
                plt.colorbar()
            
                plt.savefig(f'{plot_dir}/{args.x_constraint}-{args.y_constraint}-{args.z_constraint}-{i}-{b}.png')
                plt.close()
            ############################### 
            est_err = ((args.rescale*est).clamp(min=-20) - (args.rescale*tar).clamp(min=-20)).abs()[:,:,:-1].mean(0).mean(-1)
            int_err = ((args.rescale*interp).clamp(min=-20) - (args.rescale*tar).clamp(min=-20)).abs()[:,:,:-1].mean(0).mean(-1)
            #est_err = (args.rescale*est - args.rescale*tar).abs()[:,:,:-1].mean(0).mean(-1)
            #int_err = (args.rescale*interp - args.rescale*tar).abs()[:,:,:-1].mean(0).mean(-1)
            ############################### 
            est_err = est_err.detach().cpu().numpy()
            int_err = int_err.detach().cpu().numpy()
            est_ERR.append(est_err)
            int_ERR.append(int_err)
    with open(log_path,'a') as log:
        log.write(f"[Test] RMSE: {sum(RMSE)/(len(testset)*sel_grid)} dB\n")
        log.write(f"       SD  : {sum(SD)/(len(testset)*sel_grid)} dB\n")
    print(f"[Test] RMSE: {sum(RMSE)/(len(testset)*sel_grid)} dB")
    print(f"       SD  : {sum(SD)/(len(testset)*sel_grid)} dB")
    est_ERR = sum(est_ERR) / len(testset)
    int_ERR = sum(int_ERR) / len(testset)
    np.save(f'{plot_dir}/est_Err-{args.x_constraint}-{args.y_constraint}-{args.z_constraint}.npy', est_ERR)
    np.save(f'{plot_dir}/int_Err-{args.x_constraint}-{args.y_constraint}-{args.z_constraint}.npy', int_ERR)

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
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of inference')
    parser.add_argument('--k_folds', type=int, default=5, help='number of folds for cross-validation')
    parser.add_argument('--test_fold', type=int, default=5, help='k for test')
    parser.add_argument('--p_range', default=0.3, type=float, help='patch range')
    parser.add_argument('--rescale', default=50, type=float, help='rescale factor for input')
    parser.add_argument('--condition', type=str, default='hyper')
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--x_constraint', default=None, type=float)
    parser.add_argument('--y_constraint', default=None, type=float)
    parser.add_argument('--z_constraint', default=None, type=float)
    parser.add_argument('--film_dim', type=str, default='freq')
    args = parser.parse_args()

    args.ckpt = f'results/{args.exp_name}/train/ckpt/{args.epoch}/{args.model}_{args.step}.pt'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    recon(args)
 
