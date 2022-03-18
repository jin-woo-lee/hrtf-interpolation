import torch
import numpy as np 
from torch.utils import data
from torch.utils.data import DataLoader
import os 
import soundfile as sf 
import pickle
import glob
import scipy
import random
import librosa
from utils import load_dict, save_dict, sorted_choices
from tqdm import tqdm
import logging
import sofa

class GenericDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            directory, subj_list,
            method='S', ear='L',
            patch_range=0.2,
            n_samples=1,
            sort_by_dist=False,
        ):

        np.random.seed(0)
        logging.info(f"load dataset: {directory}")
        self.method = method     # method \in ['S', 'M']
        self.ear = ear           # ear \in ['L', 'R']
        self.data_path = sorted(glob.glob(os.path.join(directory,'*.pkl')))
        self.num_grid = 1730 if method=='S' else 440
        self.files = subj_list

        self.pos = 's_pos' if method=='S' else 'm_pos'
        self.range = patch_range
        self.n_samples = n_samples
        self.sort_by_dist = sort_by_dist
        if os.path.exists(f"nbhd/nbhd-{patch_range}.pkl"):
            self.nbhd = load_dict(f"nbhd/nbhd-{patch_range}.pkl")
            print(f"... loaded nbhd dict of range {patch_range}")
        else:
            self.prep_nbhd()
            save_dict(self.nbhd,f"nbhd/nbhd-{patch_range}.pkl")
        self.data = {}
        for data_idx in self.files:
            data_tgt = self.data_path[data_idx]
            self.data[data_idx] = load_dict(data_tgt)

    def __len__(self):
        return self.num_grid * len(self.files)

    def __getitem__(self, index):

        subj_idx = int(index // self.num_grid)
        tgts_idx = int(index % self.num_grid)
        data_idx = self.files[subj_idx]
        hrirs = self.data[data_idx]['feature'][self.ear][self.method]
        coord = self.data[data_idx]['feature'][self.ear][self.pos]
        e_l = self.data[data_idx]['label'][self.ear]['feature']
        #e_m = data['label'][self.ear]['metric']
        #t_l = data['label']['T']['feature']
        #t_m = data['label']['T']['metric']

        tar_pos = coord[tgts_idx]
        target = np.expand_dims(hrirs[tgts_idx], axis=0)
        tar_pos = np.expand_dims(tar_pos, axis=0)
        inputs = []
        src_pos = []
        if self.sort_by_dist:
            srcs_idx = sorted_choices(p=tgts_idx, qs=self.nbhd[tgts_idx], k=self.n_samples, p_sys=coord)
        else:
            srcs_idx = random.choices(self.nbhd[tgts_idx], k=self.n_samples)
        if len(srcs_idx) < self.n_samples:
            srcs_idx = random.choices(self.nbhd[tgts_idx], k=self.n_samples)
        for si in srcs_idx:
            inputs.append(np.expand_dims(hrirs[si], axis=0))
            src_pos.append(np.expand_dims(coord[si], axis=0))
        inputs = np.concatenate(inputs, axis=0)
        src_pos = np.concatenate(src_pos, axis=1)
        measure = np.expand_dims(np.array(e_l), axis=0)

        return inputs, target, measure, src_pos, tar_pos

    def prep_nbhd(self):
        subj_idx = 0
        data_idx = self.files[subj_idx]
        data_tgt = self.data_path[data_idx]
        data = load_dict(data_tgt)
        self.nbhd = {}
        print("[Loader] Gathering neighborhood info")
        for srcs_idx in range(self.num_grid):
            self.nbhd[srcs_idx] = []
        for srcs_idx in tqdm(range(self.num_grid)):
            coord = data['feature'][self.ear][self.pos]
            src_pos = coord[srcs_idx]
            for tgts_idx in range(srcs_idx,self.num_grid):
                tar_pos = coord[tgts_idx]
                dist = np.sqrt(np.sum((src_pos - tar_pos)**2))
                if dist < self.range and dist > 0:
                    if not (tgts_idx in self.nbhd[srcs_idx]):
                        self.nbhd[srcs_idx].append(tgts_idx)
                    if not (srcs_idx in self.nbhd[tgts_idx]):
                        self.nbhd[tgts_idx].append(srcs_idx)


class Trainset(GenericDataset):

    def __init__(
            self,
            directory, subj_list,
            method='S', ear='L',
            patch_range=0.2,
            n_samples=1,
            sort_by_dist=False,
        ):
        super().__init__(
            directory, subj_list,
            method=method, ear=ear, patch_range=patch_range,
            n_samples=n_samples, sort_by_dist=sort_by_dist,
        )
        print(f"[Loader] Train subject IDs:\t {self.files}")

class Testset(GenericDataset):

    def __init__(
            self,
            directory, subj_list,
            method='S', ear='L',
            patch_range=0.2, mode='Test',
            n_samples=1,
            sort_by_dist=False,
        ):
        super().__init__(
            directory, subj_list,
            method=method, ear=ear, patch_range=patch_range,
            n_samples=n_samples, sort_by_dist=sort_by_dist,
        )
        print(f"[Loader] {mode} subject IDs:\t {self.files}")

class ConstrainedSet(Testset):

    def __init__(
            self,
            directory, subj_list,
            method='S', ear='L',
            patch_range=0.2, mode='Test',
            n_samples=1,
            sort_by_dist=False,
            x_constraint=None,
            y_constraint=None,
            z_constraint=None,
        ):
        super().__init__(
            directory, subj_list,
            method=method, ear=ear, patch_range=patch_range, mode=mode,
            n_samples=n_samples, sort_by_dist=sort_by_dist,
        )
        self.sel_grid = 0
        self.selected = []
        self.select_index(x_constraint,y_constraint,z_constraint)
        self.reorder_index()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        subj_idx = int(index)
        inputs_col = []
        target_col = []
        measure_col = []
        src_pos_col = []
        tar_pos_col = []
        for tgts_idx in self.selected:
            data_idx = self.files[subj_idx]
            data_tgt = self.data_path[data_idx]
            data = load_dict(data_tgt)

            hrirs = data['feature'][self.ear][self.method]
            coord = data['feature'][self.ear][self.pos]
            e_l = data['label'][self.ear]['feature']

            tar_pos = coord[tgts_idx]
            inputs = []
            src_pos = []
            if self.sort_by_dist:
                srcs_idx = sorted_choices(p=tgts_idx, qs=self.nbhd[tgts_idx], k=self.n_samples, p_sys=coord)
            else:
                srcs_idx = random.choices(self.nbhd[tgts_idx], k=self.n_samples)
            for si in srcs_idx:
                inputs.append(np.expand_dims(hrirs[si], axis=0))
                src_pos.append(np.expand_dims(coord[si], axis=0))
            inputs = np.concatenate(inputs, axis=0)            # (in_ch, 256)
            target = np.expand_dims(hrirs[tgts_idx], axis=0)   # (1, 256)
            measure = np.expand_dims(np.array(e_l), axis=0)    # (1, 12)
            src_pos = np.concatenate(src_pos, axis=1)          # (1, 3*in_ch)
            tar_pos = np.expand_dims(tar_pos, axis=0)          # (1, 3)

            inputs_col.append(np.expand_dims(inputs, axis=0))
            target_col.append(np.expand_dims(target, axis=0))
            measure_col.append(np.expand_dims(measure, axis=0))
            src_pos_col.append(np.expand_dims(src_pos, axis=0))
            tar_pos_col.append(np.expand_dims(tar_pos, axis=0))
        inputs_col  = np.concatenate(inputs_col, axis=0)    # (sel_grid, in_ch, 256)
        target_col  = np.concatenate(target_col, axis=0)    # (sel_grid, 1, 256)
        measure_col = np.concatenate(measure_col, axis=0)   # (sel_grid, 1, 12)
        src_pos_col = np.concatenate(src_pos_col, axis=0)   # (sel_grid, 1, 3*in_ch)
        tar_pos_col = np.concatenate(tar_pos_col, axis=0)   # (sel_grid, 1, 3)
        return inputs_col, target_col, measure_col, src_pos_col, tar_pos_col

    def select_index(self, xc, yc, zc):
        data = load_dict(self.data_path[0])
        coord = data['feature'][self.ear][self.pos]
        def satisfied_constraint(x,y,z,xc,yc,zc):
            cond = True
            if (xc is not None) and round(x,5)!=xc:
                cond = False
            if (yc is not None) and round(y,5)!=yc:
                cond = False
            if (zc is not None) and round(z,5)!=zc:
                cond = False
            return cond
        for i in range(self.num_grid):
            x, y, z = coord[i]
            if satisfied_constraint(x,y,z,xc,yc,zc):
                self.selected.append(i)
        self.sel_grid = len(self.selected)


    def reorder_index(self):
        d = {}
        data = load_dict(self.data_path[0])
        coord = data['feature'][self.ear][self.pos]
        for i in range(self.sel_grid):
            x, y, z = coord[self.selected[i]]
            r = x if self.yc is not None else y
            az = np.arctan2(x,y)
            el = np.arctan2(z,r)
            az += 2*np.pi if az < 0 else 0
            el += 2*np.pi if el < 0 else 0
            ang = az if self.zc is not None else el
            if ang in d.keys():
                ang += 1e-2 * np.random.random()
            d[ang] = self.selected[i]
        od = dict(sorted(d.items()))
        i = 0
        for key, val in od.items():
            self.selected[i] = val
            i += 1

class CoarseSet(torch.utils.data.Dataset):

    def __init__(
            self,
            name,
            directory=None,
            subj_list=None,
            method='S', ear='L',
            patch_range=0.2,
            num_grid = 1730,
            n_samples=1,
            sort_by_dist=False,
            x_constraint=None,
            y_constraint=None,
            z_constraint=None,
            scale_factor=1,
        ):

        self.num_grid=num_grid
        self.method = method     # method \in ['S', 'M']
        self.ear = ear           # ear \in ['L', 'R']
        self.num_grid = num_grid
        if directory is not None:
            self.data_path = sorted(glob.glob(os.path.join(directory,'*.pkl')))
        if subj_list is not None:
             self.files = subj_list
        self.scale_factor=scale_factor
        self.xc = x_constraint
        self.yc = y_constraint
        self.zc = z_constraint
        self.src_sel_grid = 0
        self.tar_sel_grid = 0
        self.src_selected = []
        self.tar_selected = []
        self.range = patch_range
        self.n_samples = n_samples
        self.sort_by_dist = sort_by_dist

        self.pos = 's_pos' if method=='S' else 'm_pos'
        if directory is not None:
            _data = load_dict(self.data_path[0])
            self.coord = _data['feature'][self.ear][self.pos]
            self.select_index()
            self.prep_nbhd()
            mins = np.inf
            maxs = 0
            ids = self.nbhd.keys()
            for i in ids:
                lens = len(self.nbhd[i])
                if lens < mins:
                    mins = lens
                if lens > maxs:
                    maxs = lens
            self.reorder_index()
            print(f"*** number of nbhd points in [{mins}, {maxs}]")

    def __len__(self):
        return self.tar_sel_grid * len(self.files)

    def __getitem__(self, index):

        subj_idx = int(index // self.tar_sel_grid)
        tgts_idx = self.tar_selected[int(index % self.tar_sel_grid)]
        data_idx = self.files[subj_idx]
        data_tgt = self.data_path[data_idx]
        data = load_dict(data_tgt)
        hrirs = data['feature'][self.ear][self.method]
        e_l = data['label'][self.ear]['feature']
        tar_pos = self.coord[tgts_idx]

        # load srcs
        inputs = []
        src_pos = []
        if self.sort_by_dist:
            srcs_idx = sorted_choices(p=tgts_idx, qs=self.nbhd[tgts_idx], k=self.n_samples, p_sys=self.coord)
        else:
            srcs_idx = random.choices(self.nbhd[tgts_idx], k=self.n_samples)
        d = self.n_samples - len(srcs_idx)
        if d > 0:
            srcs_idx = random.choices(self.nbhd[tgts_idx], k=self.n_samples)
        for si in srcs_idx:
            inputs.append(np.expand_dims(hrirs[si], axis=0))
            src_pos.append(np.expand_dims(self.coord[si], axis=0))

        if tgts_idx == 0:
            linear = hrirs[1]
        elif tgts_idx == self.num_grid-1:
            linear = hrirs[-1]
        else:
            linear = 0.5 * (hrirs[tgts_idx-1] + hrirs[tgts_idx+1])

        inputs = np.concatenate(inputs, axis=0)
        target = np.expand_dims(hrirs[tgts_idx], axis=0)
        linear = np.expand_dims(linear, axis=0)
        src_pos = np.concatenate(src_pos, axis=1)
        tar_pos = np.expand_dims(tar_pos, axis=0)
        measure = np.expand_dims(np.array(e_l), axis=0)
        return inputs, target, linear, measure, src_pos, tar_pos

    def select_index(self):
        criterion = []
        for i in range(self.num_grid):
            x, y, z = self.coord[i]
            ang = self.coord_to_ang(x,y,z)
            if not(ang in criterion):
                criterion.append(ang)
        criterion = criterion[0::self.scale_factor]
        def scale_selected(x,y,z):
            cond = False
            ang = self.coord_to_ang(x,y,z)
            if ang in criterion:
                cond = True
            return cond
        def satisfied_constraint(x,y,z,xc,yc,zc):
            cond = True
            if (xc is not None) and round(x,5)!=xc:
                cond = False
            if (yc is not None) and round(y,5)!=yc:
                cond = False
            if (zc is not None) and round(z,5)!=zc:
                cond = False
            return cond
        for i in range(self.num_grid):
            x, y, z = self.coord[i]
            if scale_selected(x,y,z):
                self.src_selected.append(i)
            if satisfied_constraint(x,y,z,self.xc,self.yc,self.zc):
                self.tar_selected.append(i)
        self.src_sel_grid = len(self.src_selected)
        self.tar_sel_grid = len(self.tar_selected)

    def prep_nbhd(self):
        self.nbhd = {}     # nbhd among coarse grid
        print("[Loader] Gathering neighborhood info")
        for tgts_idx in range(self.tar_sel_grid):
            self.nbhd[self.tar_selected[tgts_idx]] = []
        for tgts_idx in tqdm(range(self.tar_sel_grid)):
            tar_pos = self.coord[self.tar_selected[tgts_idx]]
            for srcs_idx in range(self.src_sel_grid):
                tgts_idx_g = self.tar_selected[tgts_idx]
                srcs_idx_g = self.src_selected[srcs_idx]
                src_pos = self.coord[srcs_idx_g]
                dist = np.sqrt(np.sum((src_pos - tar_pos)**2))
                if dist < self.range:
                    if not (srcs_idx_g in self.nbhd[tgts_idx_g]):
                        self.nbhd[tgts_idx_g].append(srcs_idx_g)

    def coord_to_ang(self,x,y,z):
        r = np.sqrt(x**2 + y**2)
        az = - np.arctan2(x,y) + np.pi/2
        el = np.arctan2(z,r)
        az += 2*np.pi if az < 0 else 0
        if (self.yc is not None) and (x < 0):
            el = np.pi - el
        if (self.xc is not None) and (y < 0):
            el = np.pi - el
        el += 2*np.pi if el < 0 else 0
        ang = az if self.zc is not None else el
        return ang

    def reorder_index(self):
        d = {}
        for i in range(self.src_sel_grid):
            x, y, z = self.coord[self.src_selected[i]]
            ang = self.coord_to_ang(x,y,z)
            if ang in d.keys():
                ang += 1e-2 * np.random.random()
            d[ang] = self.src_selected[i]
        od = dict(sorted(d.items()))
        i = 0
        for key, val in od.items():
            self.src_selected[i] = val
            i += 1
        d = {}
        for i in range(self.tar_sel_grid):
            x, y, z = self.coord[self.tar_selected[i]]
            ang = self.coord_to_ang(x,y,z)
            if ang in d.keys():
                ang += 1e-2 * np.random.random()
            d[ang] = self.tar_selected[i]
        od = dict(sorted(d.items()))
        i = 0
        for key, val in od.items():
            self.tar_selected[i] = val
            i += 1



class CoarseSofaSet(CoarseSet):

    def __init__(
            self,
            name,
            sofa_path,
            anthropometry,
            patch_range=0.2,
            n_samples=1,
            num_grid = 11950,
            sort_by_dist=False,
            x_constraint=None,
            y_constraint=None,
            z_constraint=None,
            scale_factor=1,
        ):
        super().__init__(
            name,
            num_grid=num_grid,
            patch_range=patch_range,
            n_samples=n_samples,
            sort_by_dist=sort_by_dist,
            x_constraint=x_constraint,
            y_constraint=y_constraint,
            z_constraint=z_constraint,
            scale_factor=scale_factor,
        )
        self.anth = anthropometry
        self.load_hrir(sofa_path)
        self.select_index()
        self.prep_nbhd()
        mins = np.inf
        maxs = 0
        ids = self.nbhd.keys()
        for i in ids:
            lens = len(self.nbhd[i])
            if lens < mins:
                mins = lens
            if lens > maxs:
                maxs = lens
        self.reorder_index()

    def __len__(self):
        return self.tar_sel_grid

    def __getitem__(self, index):

        e_l = self.anth
        tgts_idx = self.tar_selected[int(index % self.tar_sel_grid)]
        tar_pos = self.coord[tgts_idx]

        # load srcs
        inputs = []
        src_pos = []
        if self.sort_by_dist:
            srcs_idx = sorted_choices(p=tgts_idx, qs=self.nbhd[tgts_idx], k=self.n_samples, p_sys=self.coord)
        else:
            srcs_idx = random.choices(self.nbhd[tgts_idx], k=self.n_samples)
        d = self.n_samples - len(srcs_idx)
        if d > 0:
            srcs_idx = random.choices(self.nbhd[tgts_idx], k=self.n_samples)
        for si in srcs_idx:
            inputs.append(np.expand_dims(self.hrirs[si], axis=0))
            src_pos.append(np.expand_dims(self.coord[si], axis=0))

        if tgts_idx == 0:
            linear = self.hrirs[1]
        elif tgts_idx == self.num_grid-1:
            linear = self.hrirs[-1]
        else:
            linear = 0.5 * (self.hrirs[tgts_idx-1] + self.hrirs[tgts_idx+1])

        inputs = np.concatenate(inputs, axis=0)
        target = np.expand_dims(self.hrirs[tgts_idx], axis=0)
        linear = np.expand_dims(linear, axis=0)
        src_pos = np.concatenate(src_pos, axis=1)
        tar_pos = np.expand_dims(tar_pos, axis=0)
        measure = np.expand_dims(np.array(e_l), axis=0)
        return inputs, target, linear, measure, src_pos, tar_pos

    def load_hrir(self, sofa_path):
        _file = sofa.Database.open(sofa_path)
        self.coord = _file.Source.Position.get_values(indices={"M":slice(self.num_grid)}, system="cartesian")
        self.hrirs = _file.Data.IR.get_values(indices={"M":slice(self.num_grid), "R":0, "E":0})


