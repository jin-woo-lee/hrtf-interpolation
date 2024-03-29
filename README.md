# Global HRTF Interpolation

- This repository provides overall framework for training and evaluating head-related transfer function (HRTF) interpolation systems proposed in ['Global HRTF Interpolation via Learned Affine Transformation of Hyper-conditioned Features'](https://arxiv.org/abs/2204.02637)
- Demo sound samples are available in ['here'](https://bit.ly/3DdmPu9)


### Prepare dataset

This process will re-arrange the original Sofa files, and save them in `.pkl` format. Set default arguments for `load_dir` and `save_dir` of `prep_hrir` in `dataset/preprocess.py` by "path to your HUTUBS dataset" (which contains `HRIRs` and `Antrhopometric_measures` directory) and "path to the save directory", respectively.

```bash
python3 dataset/preprocess.py
```

### Interpolation

Reconstruction test using HUTUBS. This will reproduce the results of Table 2. Note that Table 2 of the paper shows the RMSE averaged for five test folds.

```bash
python3 interp-hutubs.py --gpu 0 --test_fold 3                   # Table 2, 'All'
python3 interp-hutubs.py --gpu 0 --test_fold 3 --x_constraint 0  # Table 2, 'Fro'
python3 interp-hutubs.py --gpu 0 --test_fold 3 --y_constraint 0  # Table 2, 'Med'
python3 interp-hutubs.py --gpu 0 --test_fold 3 --z_constraint 0  # Table 2, 'Hor'
```

Interpolation test using FABIAN. This will reproduce the results of Table 3. Note that Table 3 of the paper shows the RMSE averaged for five test folds.

```bash
python3 interp-fabian.py --gpu 0 --test_fold 5                   # Table 3, Ours, 'All'
python3 interp-fabian.py --gpu 0 --test_fold 5 --y_constraint 0  # Table 3, Ours, 'Med'
python3 interp-fabian.py --gpu 0 --test_fold 5 --scale_factor 6  # Table 3, Ours (x1/6), 'All'
```

### Super-resolution

To reproduce the results of Figure 4, please see our [super-resolution tutorial](https://github.com/jin-woo-lee/hrtf-interpolation/blob/main/tutorial/FABIAN_super_resolution.ipynb) Colab notebook.

### Train

Training procedures can be found under `results` directory. Specify the path to your `HUTUBS` data directory for argument `--data_dir` (should be the same value as `save_dir` when preprocessing with `dataset/preprocess.py`).

```bash
python3 main.py --gpus 0,1 --train --cnn_layers 5 --condition hyper --in_ch 16 --p_range 0.2 --test_fold 5 --data_dir $path_to_data_dir
```

### Citation

If you find our work helpful, please cite it as below.

```bib
@inproceedings{lee2023global,
  title={Global hrtf interpolation via learned affine transformation of hyper-conditioned features},
  author={Lee, Jin Woo and Lee, Sungho and Lee, Kyogu},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```


