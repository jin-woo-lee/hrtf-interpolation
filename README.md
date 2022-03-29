# Global HRTF Interpolation

- This repository provides overall framework for training and evaluating head-related transfer function (HRTF) interpolation systems proposed in ['Global HRTF Interpolation via Learned Affine Transformation of Hyper-conditioned Features'](https://arxiv.org/abs/)


### Prepare dataset

Prepare HUTUBS dataset. This process will re-arrange the original Sofa files, and save them in `.pkl` format. Set default arguments for `load_dir` and `save_dir` of `prep_hrir` in `dataset/preprocess.py` by "path to your HUTUBS dataset" (which contains `HRIRs` and `Antrhopometric_measures` directory) and "path to the save directory", respectively.

```bash
python3 dataset/preprocess.py
```

### Evaluate

Reconstruction test for HUTUBS. This will reproduce the results of Table 2. Note that Table 2 of the paper shows the RMSE averaged for five test folds.

```bash
python3 interp-hutubs.py --gpu 0 --test_fold 3                   # Table 2, 'All'
python3 interp-hutubs.py --gpu 0 --test_fold 3 --x_constraint 0  # Table 2, 'Fro'
python3 interp-hutubs.py --gpu 0 --test_fold 3 --y_constraint 0  # Table 2, 'Med'
python3 interp-hutubs.py --gpu 0 --test_fold 3 --z_constraint 0  # Table 2, 'Hor'
```

Interpolation test for FABIAN. This will reproduce the results of Table 3. Note that Table 3 of the paper shows the RMSE averaged for five test folds.

```bash
python3 interp-fabian.py --gpu 0 --test_fold 5                   # Table 3, Ours, 'All'
python3 interp-fabian.py --gpu 0 --test_fold 5 --y_constraint 0  # Table 3, Ours, 'Med'
python3 interp-fabian.py --gpu 0 --test_fold 5 --scale_factor 6  # Table 3, Ours (x1/6), 'All'
```

### Train

Train the compensator. Training procedures can be found under `results` directory. Specify the path to your `HUTUBS` data directory for argument `--data_dir` (should be the same value as `save_dir` when preprocessing with `dataset/preprocess.py`).

```bash
python3 main.py --gpus 0,1 --train --cnn_layers 5 --condition hyper --in_ch 16 --p_range 0.2 --test_fold 5 --data_dir $path_to_data_dir
```
