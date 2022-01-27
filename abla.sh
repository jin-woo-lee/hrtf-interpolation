#!/bin/bash
#------------------------------ 
# Baseline
#------------------------------ 
#echo -------------------------------------------------- 
#echo linear
#echo -------------------------------------------------- 
#python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 0 --rnn_layers 0 --p_range 0.3 --test_fold 5 --port 70110
#python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 0 --rnn_layers 0 --p_range 0.3 --test_fold 4 --port 70111
#python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 0 --rnn_layers 0 --p_range 0.3 --test_fold 3 --port 70112
#python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 0 --rnn_layers 0 --p_range 0.3 --test_fold 2 --port 70113
#python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 0 --rnn_layers 0 --p_range 0.3 --test_fold 1 --port 70114
#echo -------------------------------------------------- 
#echo linear + chan FiLM
#echo -------------------------------------------------- 
#python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim chan --condition none --p_range 0.3 --test_fold 5 --port 70110
#python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim chan --condition none --p_range 0.3 --test_fold 4 --port 70111
#python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim chan --condition none --p_range 0.3 --test_fold 3 --port 70112
#python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim chan --condition none --p_range 0.3 --test_fold 2 --port 70113
#python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim chan --condition none --p_range 0.3 --test_fold 1 --port 70114
##echo -------------------------------------------------- 
##echo linear + freq FiLM
##echo -------------------------------------------------- 
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim freq --condition none --p_range 0.3 --test_fold 5 --port 70110
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim freq --condition none --p_range 0.3 --test_fold 4 --port 70111
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim freq --condition none --p_range 0.3 --test_fold 3 --port 70112
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim freq --condition none --p_range 0.3 --test_fold 2 --port 70113
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim freq --condition none --p_range 0.3 --test_fold 1 --port 70114
##echo -------------------------------------------------- 
##echo linear + freq FiLM + FiLM cond
##echo -------------------------------------------------- 
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim freq --condition film --p_range 0.3 --test_fold 5 --port 70110
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim freq --condition film --p_range 0.3 --test_fold 4 --port 70111
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim freq --condition film --p_range 0.3 --test_fold 3 --port 70112
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim freq --condition film --p_range 0.3 --test_fold 2 --port 70113
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim freq --condition film --p_range 0.3 --test_fold 1 --port 70114
##echo -------------------------------------------------- 
##echo linear + freq FiLM + hyperconv
##echo -------------------------------------------------- 
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim freq --condition hyper --p_range 0.3 --test_fold 5 --port 70110
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim freq --condition hyper --p_range 0.3 --test_fold 4 --port 70111
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim freq --condition hyper --p_range 0.3 --test_fold 3 --port 70112
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim freq --condition hyper --p_range 0.3 --test_fold 2 --port 70113
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim freq --condition hyper --p_range 0.3 --test_fold 1 --port 70114
##echo -------------------------------------------------- 
##echo linear + freq FiLM + hyperconv + rnn
##echo -------------------------------------------------- 
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 5 --film_dim freq --condition hyper --p_range 0.3 --test_fold 5 --port 70110
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 5 --film_dim freq --condition hyper --p_range 0.3 --test_fold 4 --port 70111
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 5 --film_dim freq --condition hyper --p_range 0.3 --test_fold 3 --port 70112
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 5 --film_dim freq --condition hyper --p_range 0.3 --test_fold 2 --port 70113
##python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 5 --film_dim freq --condition hyper --p_range 0.3 --test_fold 1 --port 70114

echo -------------------------------------------------- 
echo linear + chan FiLM + hyperconv
echo -------------------------------------------------- 
python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim chan --condition hyper --p_range 0.3 --test_fold 5 --port 70210
python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim chan --condition hyper --p_range 0.3 --test_fold 4 --port 70211
python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim chan --condition hyper --p_range 0.3 --test_fold 3 --port 70212
python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim chan --condition hyper --p_range 0.3 --test_fold 2 --port 70213
python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim chan --condition hyper --p_range 0.3 --test_fold 1 --port 70214
echo -------------------------------------------------- 
echo linear + chan FiLM + film
echo -------------------------------------------------- 
python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim chan --condition film --p_range 0.3 --test_fold 5 --port 70110
python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim chan --condition film --p_range 0.3 --test_fold 4 --port 70111
python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim chan --condition film --p_range 0.3 --test_fold 3 --port 70112
python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim chan --condition film --p_range 0.3 --test_fold 2 --port 70113
python3 main.py --gpus 12 --test t --load_epoch 100 --in_ch 8 --cnn_layers 5 --rnn_layers 0 --film_dim chan --condition film --p_range 0.3 --test_fold 1 --port 70114
