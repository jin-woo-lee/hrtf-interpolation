import os
import csv
import glob
import sofa
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pickle

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def retrieve_elevation(arr, z_coord=0.):
    idx = []
    for i, p in enumerate(arr):
        if p[2] == z_coord:
            idx.append(i)
    return np.array(idx)

def get_z_coords(arr):
    idx = []
    for i, p in enumerate(arr):
        if not (p[2] in idx):
            idx.append(p[2])
    return np.array(idx)

def get_antrhopometry_dict(am_dir):
    am = []
    with open(os.path.join(am_dir, 'AntrhopometricMeasures.csv'), newline='') as csvfile:
        metric = None
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == 0:
                metric = row
                continue
            elif any(t == 'NaN' for t in row):
                continue
            else:
                # for each subject
                dict_ = {
                    'T': {},    # torso
                    'L': {},    # left ear
                    'R': {},    # right ear
                }
                feat_t = []
                feat_l = []
                feat_r = []
                for j, l in enumerate(metric):
                    if j==0:
                        dict_[l] = int(row[j])
                    elif 'x' in l:
                        feat_t.append(float(row[j]))
                    elif 'L' in l:
                        feat_l.append(float(row[j]))
                    elif 'R' in l:
                        feat_r.append(float(row[j]))
                    else:
                        print(row[j])
                dict_['T']['metric'] = metric[1:14]
                dict_['L']['metric'] = metric[14:26]
                dict_['R']['metric'] = metric[26:38]
                dict_['T']['feature'] = feat_t
                dict_['L']['feature'] = feat_l
                dict_['R']['feature'] = feat_r
                assert  len(dict_['T']['metric'])==len(dict_['T']['feature'])\
                    and len(dict_['L']['metric'])==len(dict_['L']['feature'])\
                    and len(dict_['R']['metric'])==len(dict_['R']['feature']),\
                    'data length should match' 
                am.append(dict_)
    return am


def get_hrir_data(path,
                 flag='simulated', ear='L', elevation=0,
                 emitter=0):
    # flag \in ['simulated', 'measured']
    # ear \in ['L', 'R']
    measure_pts = 1730 if flag=='simulated' else 440
    receiver = 0 if ear=='L' else 1
    HRIR = sofa.Database.open(path)
    #HRIR.Metadata.dump()
    pos = np.round(HRIR.Source.Position.get_values(indices={"M":slice(measure_pts)}, system="cartesian"),2)
    ############################### 
    # full elevation
    #idx = retrieve_elevation(pos, z_coord=elevation)
    idx = np.arange(0,measure_pts,1)
    ############################### 
    
    ir_data = HRIR.Data.IR.get_values(indices={"M":idx, "R":receiver, "E":emitter})
    #print(ir_data.shape)    # (measure_pts, 256)
    HRIR.close()
    return ir_data

def get_position(path, flag='simulated'):
    measure_pts = 1730 if flag=='simulated' else 440
    HRIR = sofa.Database.open(path)
    pos = HRIR.Source.Position.get_values(indices={"M":slice(measure_pts)}, system="cartesian")
    return np.array(pos)

def struct_data(root_dir='/data2/HRTF/HUTUBS/HUTUBS', elevation=0.):
    # get output label
    am_dir = os.path.join(root_dir, 'Antrhopometric_measures')
    AM = get_antrhopometry_dict(am_dir)
    # AM = [
    # { 'SubjectID': (int) subject id
    #   'T': {'metric': (list) antrhopometry names of torso,     'feature': (list) antrhopometries of torso}
    #   'L': {'metric': (list) antrhopometry names of left ear,  'feature': (list) antrhopometries of left ear}
    #   'R': {'metric': (list) antrhopometry names of right ear, 'feature': (list) antrhopometries of right ear}
    # }, ... ]

    # get input data
    ir_dir = os.path.join(root_dir, 'HRIRs')
    ir_sim = sorted(glob.glob(os.path.join(ir_dir, '*_simulated.sofa')))
    ir_mes = sorted(glob.glob(os.path.join(ir_dir, '*_measured.sofa')))
    assert len(ir_sim)==len(ir_mes), 'HRIR simulation and measure data match'
    HRIR = []
    for i in range(len(ir_sim)):
        dict_ = {}
        ids = int(ir_sim[i].split('/')[-1].split('_')[0].split('pp')[-1])
        idm = int(ir_mes[i].split('/')[-1].split('_')[0].split('pp')[-1])
        assert ids == idm, 'IDs of simulated and measured data should match'
        dict_['SubjectID'] = ids
        hrir_L = {}
        hrir_R = {}
        ############################### 
        # full elevation
        #hrir_L['S'] = get_hrir_data(ir_sim[i], ear='L', elevation=elevation)
        #hrir_L['M'] = get_hrir_data(ir_mes[i], ear='L', elevation=elevation)
        #hrir_R['S'] = get_hrir_data(ir_sim[i], ear='R', elevation=elevation)
        #hrir_R['M'] = get_hrir_data(ir_mes[i], ear='R', elevation=elevation)
        hrir_L['S'] = get_hrir_data(ir_sim[i], ear='L', flag='simulated')
        hrir_L['M'] = get_hrir_data(ir_mes[i], ear='L', flag='measured')
        hrir_R['S'] = get_hrir_data(ir_sim[i], ear='R', flag='simulated')
        hrir_R['M'] = get_hrir_data(ir_mes[i], ear='R', flag='measured')
        sim_pos = get_position(ir_sim[i], 'simulated')
        mes_pos = get_position(ir_mes[i], 'measured')
        hrir_L['s_pos'] = sim_pos
        hrir_L['m_pos'] = mes_pos
        hrir_R['s_pos'] = sim_pos
        hrir_R['m_pos'] = mes_pos
        ############################### 
        dict_['L'] = hrir_L
        dict_['R'] = hrir_R
        HRIR.append(dict_)
    # HRIR = [
    # { 'SubjectID': (int) subject id
    #   'simulated': (np.array) simulated HRIR
    #   'measured': (np.array) measured HRIR
    # }, ... ]

    DATA = []
    n_subj = min(len(AM), len(HRIR))
    for i in range(n_subj):
        dict_ = {}
        am = AM[i]
        ida = am['SubjectID']
        found = False
        for j in range(len(HRIR)):
            if HRIR[j]['SubjectID'] == ida:
                hrir = HRIR[j]
                found = True
                break
        if not found:
            raise IndexError("Subject ID not matched")
        dict_['SubjectID'] = ida
        dict_['feature'] = {
            'L': hrir['L'],
            'R': hrir['R'],
        }
        dict_['label'] = {
            'T': am['T'],
            'L': am['L'],
            'R': am['R'],
        }
        DATA.append(dict_)
    return DATA

def prep_hrir(load_dir='/data2/HRTF/HUTUBS/HUTUBS', save_dir='/data2/HRTF/HUTUBS/pkl-15', elevation=0.):
    print("Load dataset", load_dir)
    DATA = struct_data(load_dir,elevation)
    print("Save to", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    for data in DATA:
        id_ = data['SubjectID']
        name = f'{id_}.pkl'
        path = os.path.join(save_dir, name)
        save_dict(data, path)
    print("done")

if __name__=='__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ))
    from utils import load_dict
    """
    DATA = [
        {
            'SubjectID': (int) subject id
            'feature': {
                'L': {
                    'S': (np.array) simulated HRIR for left ear
                    'M': (np.array) measured HRIR for left ear
                    's_pos': (np.array) coordinates of simulation point grid
                    'm_pos': (np.array) coordinates of measurement point grid
                }
                'R': {
                    'S': (np.array) simulated HRIR for right ear
                    'M': (np.array) measured HRIR for right ear
                    's_pos': (np.array) coordinates of simulation point grid
                    'm_pos': (np.array) coordinates of measurement point grid
                }
            }
            'label': {
                'T': {
                    'metric': (list) antrhopometry names of torso
                    'feature': (list) antrhopometries of torso
                }
                'L': {
                    'metric': (list) antrhopometry names of left ear
                    'feature': (list) antrhopometries of left ear
                }
                'R': {
                    'metric': (list) antrhopometry names of right ear
                    'feature': (list) antrhopometries of right ear
                }
            }
        }, ... ]
    """
    prep_hrir()
    #x = load_dict('/data2/HUTUBS/HRIRs/1.pkl')
    #print(x['feature']['L']['S'].shape)


