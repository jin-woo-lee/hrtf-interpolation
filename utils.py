import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import pickle
import torch.nn.functional as F
import collections

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def minmax_normalize(x):
    b, c, f, t = x.shape
    x_min = x.reshape(b,c*f*t).min(dim=-1).values.reshape(b,1,1,1)
    x = x - x_min
    x_max = x.reshape(b,c*f*t).max(dim=-1).values.reshape(b,1,1,1)
    x = x / x_max
    return x

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def adjust_noise(noise, source, snr, divide_by_max = True):
    eps = np.finfo(np.float32).eps
    noise_rms = cal_rms(noise) # noise rms

    num = cal_rms(source) # source rms
    den = np.power(10., snr/20)
    desired_noise_rms = num/den

    # calculate gain
    try:
        gain = desired_noise_rms / (noise_rms + eps)
    except OverflowError:
        gain = 1.

    noise = gain*noise

    mix = source + noise

    if divide_by_max == True:
        mix_max_val = np.abs(mix).max(axis=-1)
        src_max_val = np.abs(source).max(axis=-1)
        noise_max_val = np.abs(noise).max(axis=-1)

        if (mix_max_val > 1.) or (src_max_val > 1.) or (noise_max_val > 1.):
            max_val = np.max([mix_max_val, src_max_val, noise_max_val])
            mix = mix / (max_val+eps)
            source = source / (max_val+eps)
            noise = noise / (max_val+eps)
        else:
            pass
    else:
        pass

    return mix, source, noise

def rms_normalize(wav, ref_dB=-23.0):
    # RMS normalize
    eps = np.finfo(np.float32).eps
    rms = cal_rms(wav)
    rms_dB = 20*np.log(rms/1) # rms_dB
    ref_linear = np.power(10, ref_dB/20.)
    gain = ref_linear / np.sqrt(np.mean(np.square(wav), axis=-1) + eps)
    wav = gain * wav
    return wav

def pos_enc(lens,coord):
    pos = np.arange(lens) / lens

    elevation = np.arcsin(coord[2]/1.47)
    azimuth = - np.arctan(coord[1]/coord[0])
    if coord[0] >= 0:
        if coord[1] >= 0:
            azimuth += np.pi
        elif coord[0] > 0:
            azimuth -= np.pi
    e_gain, a_gain = 1, 1
    if azimuth < 0:
        a_gain *= -1
    #print(f"[{coord[0]},\t{coord[1]}]:\t {azimuth}")

    penc = -e_gain * np.sin(40*elevation*pos)
    penc += a_gain * np.cos(200*azimuth*pos)
    return np.expand_dims(penc, axis=0)

def plot_sample(tf_data, dirs, pos, test=None):
    batch_sz = len(tf_data[0])
    freqs = np.linspace(0,22050,tf_data[0].shape[-1])
    #for b in range(batch_sz):
    for b in range(1):
        title=f"target pos [{pos[b][0]}, {pos[b][1]}, {pos[b][1]}]"
        tf_maxs = [tf_data[j][b].max() for j in range(len(tf_data))]
        tf_mins = [tf_data[j][b].min() for j in range(len(tf_data))]
        tf_max = max(tf_maxs)+1
        tf_min = min(tf_mins)-1
        npl = 210
        figure = plt.figure(figsize=(13,7))
        plt.suptitle(title)
        #---------- 
        plt.subplot(npl+1)
        in_ch = tf_data[0].shape[1]
        for i in range(in_ch):
            plt.plot(freqs, tf_data[0][b,i], label=f'input-{i}', color='k', linewidth=.3)
        plt.plot(freqs, tf_data[1][b,0], label='target', color='b', linewidth=.8)
        plt.plot(freqs, tf_data[2][b,0], label='estimate', color='r', linewidth=.8)
        plt.ylim(tf_min,tf_max)
        #plt.legend()
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (dB)")
        plt.axhline(y=0, color='k', linewidth=.5)
        plt.tight_layout()
        #---------- 
        diff = tf_data[2][b,0] - tf_data[1][b,0]
        plt.subplot(npl+2)
        plt.plot(freqs, diff, label='difference', color='r', linewidth=.8)
        plt.plot(freqs, tf_data[3][b,0], label='difference', color='b', linewidth=.8)
        plt.ylim(-10,10)
        #plt.legend()
        plt.axhline(y=0, color='k', linewidth=.5)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (dB)")
        plt.tight_layout()
        #---------- 
        if test is not None:
            plt.savefig(f"{dirs}/{test}-{b}.png")
        else:
            plt.savefig(f"{dirs}/{b}.png")
        plt.close()
    return figure

def coord_to_angle(coord, unit,eps=1e-5):
    B = len(coord)
    azim = []
    elev = []
    for b in range(B):
        el = np.arcsin(coord[b,2]/(1.47+eps))
        az = - np.arctan(coord[b,1]/coord[b,0])
        if coord[b,0] >= 0:
            if coord[b,1] >= 0:
                az += np.pi
            elif coord[b,0] > 0:
                az -= np.pi

        if unit=='deg':
            az *= (180/np.pi)
            el *= (180/np.pi)
        azim.append(np.round(az,2))
        elev.append(np.round(el,2))
    return azim, elev

def logmag_phase(ir, n_fft=256):
    spec = torch.fft.rfft(ir, n_fft)
    mag = spec.abs().clamp(min=1e-5)
    phs = spec / mag
    phs = torch.cat([phs.real.unsqueeze(-1),phs.imag.unsqueeze(-1)], dim=-1)
    logmag = 20 * mag.log10()
    return logmag, phs

def phase_loss(phs_1, phs_2):
    cossim = F.cosine_similarity(phs_1, phs_2, dim=-1)    # [-1, 1]
    ground = torch.ones_like(cossim)
    return F.l1_loss(cossim, ground)

def sorted_choices(p,qs,k,p_sys,q_sys=None):
    if q_sys is None:
        q_sys = p_sys
    pc = p_sys[p]
    d = {}
    for q in qs:
        dist = np.sqrt(np.absolute(q_sys[q] - p_sys[p]).sum())
        if dist in d.keys():
            dist += 1e-2 * np.random.random()
        d[dist] = q
    n = 0
    out = []
    od = dict(sorted(d.items()))
    for key, val in od.items():
        if n == k:
            break
        out.append(val)
        n += 1
    return out
