#!/bin/bash
#-------------------------------------------------- 
echo --------------------------------------------------
echo Scale x2
echo --------------------------------------------------
python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold5 --in_ch 8 --scale_factor 2
python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold4 --in_ch 8 --scale_factor 2
python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold3 --in_ch 8 --scale_factor 2
python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold2 --in_ch 8 --scale_factor 2
python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold1 --in_ch 8 --scale_factor 2
echo --------------------------------------------------                                                          
echo Scale x4                                                                                                    
echo --------------------------------------------------                                                          
python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold5 --in_ch 8 --scale_factor 4
python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold4 --in_ch 8 --scale_factor 4
python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold3 --in_ch 8 --scale_factor 4
python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold2 --in_ch 8 --scale_factor 4
python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold1 --in_ch 8 --scale_factor 4
echo --------------------------------------------------                                                          
echo Scale x8                                                                                                    
echo --------------------------------------------------                                                          
python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold5 --in_ch 8 --scale_factor 8
python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold4 --in_ch 8 --scale_factor 8
python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold3 --in_ch 8 --scale_factor 8
python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold2 --in_ch 8 --scale_factor 8
python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold1 --in_ch 8 --scale_factor 8
#-------------------------------------------------- 
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold5 --in_ch 8 --z_constraint 0 --scale_factor 2
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold4 --in_ch 8 --z_constraint 0 --scale_factor 2
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold3 --in_ch 8 --z_constraint 0 --scale_factor 2
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold2 --in_ch 8 --z_constraint 0 --scale_factor 2
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold1 --in_ch 8 --z_constraint 0 --scale_factor 2
#python3 analyze_error.py                                                                                         
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold5 --in_ch 8 --y_constraint 0 --scale_factor 2
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold4 --in_ch 8 --y_constraint 0 --scale_factor 2
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold3 --in_ch 8 --y_constraint 0 --scale_factor 2
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold2 --in_ch 8 --y_constraint 0 --scale_factor 2
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold1 --in_ch 8 --y_constraint 0 --scale_factor 2
#python3 analyze_error.py                                                                                         
#--------------------------------------------------                                                               
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold5 --in_ch 8 --z_constraint 0 --scale_factor 4
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold4 --in_ch 8 --z_constraint 0 --scale_factor 4
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold3 --in_ch 8 --z_constraint 0 --scale_factor 4
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold2 --in_ch 8 --z_constraint 0 --scale_factor 4
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold1 --in_ch 8 --z_constraint 0 --scale_factor 4
#python3 analyze_error.py                                                                                         
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold5 --in_ch 8 --y_constraint 0 --scale_factor 4
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold4 --in_ch 8 --y_constraint 0 --scale_factor 4
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold3 --in_ch 8 --y_constraint 0 --scale_factor 4
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold2 --in_ch 8 --y_constraint 0 --scale_factor 4
#python3 interp-hutubs.py --gpu 6 --epoch 100 --exp_name hyperfilm-5x5-Fi_freq-Co_hyper-d_0.3-N_8-5fold1 --in_ch 8 --y_constraint 0 --scale_factor 4
#python3 analyze_error.py
#-------------------------------------------------- 
