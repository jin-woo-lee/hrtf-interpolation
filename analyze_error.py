import os
import numpy as np
import matplotlib.pyplot as plt

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

#d=0.3
#epoch = 100
#film = 'freq'
#cond = 'hyper'
#fr = np.linspace(0,22050,129)
#plt.figure(figsize=(5,5))
#for N in [4,8,12,16]:
#    FE = []
#    for fold in [5,4,3,2,1]:
#        path = f'results/hyperfilm-5x5-Fi_{film}-Co_{cond}-d_{d}-N_{N}-5fold{fold}/test/{epoch}/freqs_error.npy'
#        fe = np.load(path)
#        FE.append(fe)
#    FE = sum(FE) / 5
#    plt.plot(fr,FE, label=f'N={N}')
#
#plt.xlabel('Frequency (Hz)')
#plt.ylabel('SD (dB)')
#plt.ylim([0,2])
#plt.legend()
#plt.tight_layout()
#plt.savefig('freq-study.png')

epoch = 100
film = 'freq'
cond = 'hyper'
plt.figure(figsize=(5,5))
Ns = [4,8,12,16]
ds = [0.2, 0.3, 0.4, 0.5, 0.6]
for N in Ns:
    for d in ds:
        E = []
        for fold in [5,4,3,2,1]:
            path = f'results/hyperfilm-5x5-Fi_{film}-Co_{cond}-d_{d}-N_{N}-5fold{fold}/test/{epoch}/freqs_error.npy'
            fe = np.load(path)
            E.append(fe)
        E = sum(E) / 5
        plt.plot(fr,E, label=f'delta={N}')
plt.xlabel('N')
plt.ylabel('LSD')
#plt.ylim([0,2])
plt.legend()
plt.tight_layout()
plt.savefig('freq-study.png')




#d=0.3
#N = 8
#epoch = 100
#film = 'freq'
#cond = 'hyper'
#for cons in [[None, 0.0, None],[None, None, 0.0]]:
#    xc, yc, zc = cons
#    plt.figure(figsize=(13,5))
#    est_FE = []
#    int_FE = []
#    for fold in [5,4,3,2,1]:
#        est_path = f'results/hyperfilm-5x5-Fi_{film}-Co_{cond}-d_{d}-N_{N}-5fold{fold}/test/fabian/{epoch}/plot/est_Err-{xc}-{yc}-{zc}.npy'
#        int_path = f'results/hyperfilm-5x5-Fi_{film}-Co_{cond}-d_{d}-N_{N}-5fold{fold}/test/fabian/{epoch}/plot/int_Err-{xc}-{yc}-{zc}.npy'
#        est_fe = np.load(est_path)
#        int_fe = np.load(int_path)
#        est_FE.append(est_fe)
#        int_FE.append(int_fe)
#    est_FE = sum(est_FE) / 5
#    int_FE = sum(int_FE) / 5
#    degs = len(est_FE)
#    az = np.linspace(0,360,degs)
#    plt.plot(az,est_FE, label=f'estimated')
#    plt.plot(az,int_FE, label=f'linear interp')
#    
#    if zc is not None:
#        lab = 'Azimuth'
#    elif yc is not None:
#        lab = 'Elevation'
#    
#    plt.xlabel(f'{lab} (deg)')
#    plt.ylabel('SD (dB)')
#    plt.legend()
#    plt.tight_layout()
#    plt.savefig(f'fabian-recon-{lab}-study.png')
    
 
#d=0.3
#N = 4
#epoch = 100
#film = 'freq'
#cond = 'hyper'
#for cons in [[None, 0.0, None],[None, None, 0.0]]:
#    xc, yc, zc = cons
#    plt.figure(figsize=(13,5))
#    est_FE = []
#    int_FE = []
#    for fold in [5,4,3,2,1]:
#        est_path = f'results/hyperfilm-5x5-Fi_{film}-Co_{cond}-d_{d}-N_{N}-5fold{fold}/test/fabian-interp/{epoch}/plot/est_Err-{xc}-{yc}-{zc}.npy'
#        int_path = f'results/hyperfilm-5x5-Fi_{film}-Co_{cond}-d_{d}-N_{N}-5fold{fold}/test/fabian-interp/{epoch}/plot/int_Err-{xc}-{yc}-{zc}.npy'
#        est_fe = np.load(est_path)
#        int_fe = np.load(int_path)
#        est_FE.append(est_fe)
#        int_FE.append(int_fe)
#    est_FE = sum(est_FE) / 5
#    int_FE = sum(int_FE) / 5
#    degs = len(est_FE)
#    az = np.linspace(0,360,degs)
#    plt.plot(az,est_FE, label=f'estimated')
#    plt.plot(az,int_FE, label=f'linear interp')
#    
#    if zc is not None:
#        lab = 'Azimuth'
#    elif yc is not None:
#        lab = 'Elevation'
#    
#    plt.xlabel(f'{lab} (deg)')
#    plt.ylabel('SD (dB)')
#    plt.legend()
#    plt.tight_layout()
#    plt.savefig(f'fabian-interp-{lab}-study.png')
    
  
#d=0.3
#N = 8
#epoch = 100
#film = 'freq'
#cond = 'hyper'
#scale_factor = 4
#os.makedirs('anal',exist_ok=True)
#for cons in ['hor','med']:
#    plt.figure(figsize=(13,5))
#    est_FE = []
#    int_FE = []
#    name = f'HUT_C2F_{cons}_{scale_factor}'
#    for fold in [5,4,3,2,1]:
#        est_path = f'results/hyperfilm-5x5-Fi_{film}-Co_{cond}-d_{d}-N_{N}-5fold{fold}/test/hutubs-interp/{epoch}/error/{name}-est-sd.npy'
#        int_path = f'results/hyperfilm-5x5-Fi_{film}-Co_{cond}-d_{d}-N_{N}-5fold{fold}/test/hutubs-interp/{epoch}/error/{name}-lin-sd.npy'
#        est_fe = np.load(est_path)
#        int_fe = np.load(int_path)
#        est_FE.append(est_fe)
#        int_FE.append(int_fe)
#    est_FE = sum(est_FE) / 5
#    int_FE = sum(int_FE) / 5
#    degs = len(est_FE)
#    az = np.linspace(0,360,degs)
#    plt.plot(az,est_FE, label=f'estimated')
#    plt.plot(az,int_FE, label=f'linear interp')
#    
#    if cons=='hor':
#        lab = 'Azimuth'
#    else:
#        lab = 'Elevation'
#    
#    plt.xlabel(f'{lab} (deg)')
#    plt.ylabel('SD (dB)')
#    plt.legend()
#    plt.tight_layout()
#    plt.savefig(f'anal/{name}-{lab}-study.png')
#    
#             
