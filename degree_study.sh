#!/bin/bash
#python3 interp-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_4-5fold5 --z_constraint 0 --in_ch 4
#python3 interp-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_4-5fold4 --z_constraint 0 --in_ch 4
#python3 interp-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_4-5fold3 --z_constraint 0 --in_ch 4
#python3 interp-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_4-5fold2 --z_constraint 0 --in_ch 4
#python3 interp-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_4-5fold1 --z_constraint 0 --in_ch 4
#python3 analyze_error.py
#python3 interp-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_4-5fold5 --y_constraint 0 --in_ch 4
#python3 interp-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_4-5fold4 --y_constraint 0 --in_ch 4
#python3 interp-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_4-5fold3 --y_constraint 0 --in_ch 4
#python3 interp-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_4-5fold2 --y_constraint 0 --in_ch 4
#python3 interp-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_4-5fold1 --y_constraint 0 --in_ch 4
#python3 analyze_error.py
python3 recon-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold5 --z_constraint 0 --in_ch 8
python3 recon-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold4 --z_constraint 0 --in_ch 8
python3 recon-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold3 --z_constraint 0 --in_ch 8
python3 recon-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold2 --z_constraint 0 --in_ch 8
python3 recon-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold1 --z_constraint 0 --in_ch 8
python3 analyze_error.py
python3 recon-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold5 --y_constraint 0 --in_ch 8
python3 recon-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold4 --y_constraint 0 --in_ch 8
python3 recon-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold3 --y_constraint 0 --in_ch 8
python3 recon-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold2 --y_constraint 0 --in_ch 8
python3 recon-fabian.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold1 --y_constraint 0 --in_ch 8
python3 analyze_error.py
