# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 18:51:32 2021

@author: Jinuk
"""

import scipy.io as sio
import os
from librosa.util import normalize

import torch
import torch.nn as nn
import torchaudio.transforms as AT
import noisereduce as nr

Sessions = [1, 2, 3, 4, 5, 6, 7, 8, 9]

Subs = [1]
extra = 0.2
fs = 2048
sr = 22050
num_mels = 80

win_length = 0.04
hop_length = 0.02
hop_size = round(hop_length * sr)
win_size = round(win_length * sr)
n_fft = 1024

if ((n_fft - hop_size) % 2 == 1):
    pad_size = (int((n_fft - hop_size) / 2) + 1, int((n_fft - hop_size) / 2))
else:
    pad_size = (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2))

fmin = 0
fmax = 8000

mel_basis = {}
hann_window = {}

snd = {}
mel_spectrogram = AT.MelSpectrogram(sample_rate=sr, n_fft=n_fft, win_length=win_size, hop_length=hop_size, n_mels=80,
                                    f_max=8000, center=False, onesided=True)
Amp2DB = AT.AmplitudeToDB()
for sub in Subs:
    print(sub)
    snd['snd'] = {}
    snd['mel'] = {}

    save_folder = 'data/feat/sub' + str(sub) + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    nTrial = 0
    for sess in Sessions:
        temp_snd = sio.loadmat('data/epo/sub' + str(sub) + '/snd/epo' + str(sess))
        temp = temp_snd['snd']['x'].item()
        nData = temp.shape[1]

        for dat in range(nData):
            snd_org = temp[0, dat].squeeze()

            snd_nr = nr.reduce_noise(y=snd_org, sr=sr, prop_decrease=0.95)
            snd_nr_norm = normalize(snd_nr) * 0.95
            snd['snd']['nr_norm_epo' + str(nTrial + 1)] = snd_nr_norm
            mel = torch.FloatTensor(snd_nr_norm).unsqueeze(0)
            mel = torch.nn.functional.pad(mel.unsqueeze(1), pad_size, mode='reflect').squeeze(1)
            mel = mel_spectrogram(mel)
            mel = torch.log(torch.clamp(mel, min=1e-4))
            snd['mel']['nr_epo' + str(nTrial + 1)] = mel.numpy().transpose()


            nTrial = nTrial + 1


    sio.savemat(save_folder + '/snd.mat', snd)
