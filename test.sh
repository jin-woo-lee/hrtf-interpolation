#!/bin/bash
echo ================================================== 
echo N = 4, p =  0.2
echo ================================================== 
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.2 --film_dim chan --condition hyper --test_fold 5 --port 80110
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.2 --film_dim chan --condition hyper --test_fold 4 --port 80111
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.2 --film_dim chan --condition hyper --test_fold 3 --port 80112
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.2 --film_dim chan --condition hyper --test_fold 2 --port 80113
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.2 --film_dim chan --condition hyper --test_fold 1 --port 80114
echo ================================================== 
echo N = 4, p =  0.3
echo ================================================== 
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.3 --film_dim chan --condition hyper --test_fold 5 --port 80110
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.3 --film_dim chan --condition hyper --test_fold 4 --port 80111
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.3 --film_dim chan --condition hyper --test_fold 3 --port 80112
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.3 --film_dim chan --condition hyper --test_fold 2 --port 80113
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.3 --film_dim chan --condition hyper --test_fold 1 --port 80114
echo ================================================== 
echo N = 4, p =  0.4
echo ================================================== 
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.4 --film_dim chan --condition hyper --test_fold 5 --port 80110
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.4 --film_dim chan --condition hyper --test_fold 4 --port 80111
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.4 --film_dim chan --condition hyper --test_fold 3 --port 80112
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.4 --film_dim chan --condition hyper --test_fold 2 --port 80113
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.4 --film_dim chan --condition hyper --test_fold 1 --port 80114
echo ================================================== 
echo N = 4, p =  0.5
echo ================================================== 
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.5 --film_dim chan --condition hyper --test_fold 5 --port 80110
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.5 --film_dim chan --condition hyper --test_fold 4 --port 80111
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.5 --film_dim chan --condition hyper --test_fold 3 --port 80112
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.5 --film_dim chan --condition hyper --test_fold 2 --port 80113
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.5 --film_dim chan --condition hyper --test_fold 1 --port 80114
echo ================================================== 
echo N = 4, p =  0.6
echo ================================================== 
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.6 --film_dim chan --condition hyper --test_fold 5 --port 80110
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.6 --film_dim chan --condition hyper --test_fold 4 --port 80111
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.6 --film_dim chan --condition hyper --test_fold 3 --port 80112
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.6 --film_dim chan --condition hyper --test_fold 2 --port 80113
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 4 --p_range 0.6 --film_dim chan --condition hyper --test_fold 1 --port 80114


echo ================================================== 
echo N = 8, p =  0.2
echo ================================================== 
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.2 --film_dim chan --condition hyper --test_fold 5 --port 80110
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.2 --film_dim chan --condition hyper --test_fold 4 --port 80111
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.2 --film_dim chan --condition hyper --test_fold 3 --port 80112
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.2 --film_dim chan --condition hyper --test_fold 2 --port 80113
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.2 --film_dim chan --condition hyper --test_fold 1 --port 80114
echo ================================================== 
echo N = 8, p =  0.3
echo ================================================== 
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.3 --film_dim chan --condition hyper --test_fold 5 --port 80110
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.3 --film_dim chan --condition hyper --test_fold 4 --port 80111
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.3 --film_dim chan --condition hyper --test_fold 3 --port 80112
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.3 --film_dim chan --condition hyper --test_fold 2 --port 80113
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.3 --film_dim chan --condition hyper --test_fold 1 --port 80114
echo ================================================== 
echo N = 8, p =  0.4
echo ================================================== 
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.4 --film_dim chan --condition hyper --test_fold 5 --port 80110
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.4 --film_dim chan --condition hyper --test_fold 4 --port 80111
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.4 --film_dim chan --condition hyper --test_fold 3 --port 80112
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.4 --film_dim chan --condition hyper --test_fold 2 --port 80113
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.4 --film_dim chan --condition hyper --test_fold 1 --port 80114
echo ================================================== 
echo N = 8, p =  0.5
echo ================================================== 
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.5 --film_dim chan --condition hyper --test_fold 5 --port 80110
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.5 --film_dim chan --condition hyper --test_fold 4 --port 80111
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.5 --film_dim chan --condition hyper --test_fold 3 --port 80112
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.5 --film_dim chan --condition hyper --test_fold 2 --port 80113
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.5 --film_dim chan --condition hyper --test_fold 1 --port 80114
echo ================================================== 
echo N = 8, p =  0.6
echo ================================================== 
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.6 --film_dim chan --condition hyper --test_fold 5 --port 80110
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.6 --film_dim chan --condition hyper --test_fold 4 --port 80111
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.6 --film_dim chan --condition hyper --test_fold 3 --port 80112
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.6 --film_dim chan --condition hyper --test_fold 2 --port 80113
python3 main.py --gpus 12 --test t --cnn_layers 5 --rnn_layers 0 --load_epoch 100 --in_ch 8 --p_range 0.6 --film_dim chan --condition hyper --test_fold 1 --port 80114


