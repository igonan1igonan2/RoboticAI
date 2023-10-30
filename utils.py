from torch.utils.data import Dataset
import numpy as np
"""
The architecture is based on the paper “Attention Is All You Need”. 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.
"""
import glob
import random
import pyworld
import pysptk
import os
import shutil
from pystoi import stoi as stoii
from pesq import pesq as pesqq
from librosa import resample as resam
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))
        
from nnmnkwii.metrics import melcd

def cal_mcd(wav1, wav2, sr = 22050, FRAME_PERIOD = 5.0, alpha = 0.35, fft_size = 1024, mcep_dim = 25, dtw = True):
    
    _, sp1, _ = pyworld.wav2world(wav1.astype(np.double), fs=sr, frame_period=FRAME_PERIOD, fft_size=fft_size)
    mgc1 = pysptk.sptk.mcep(sp1, order=mcep_dim, alpha=alpha, maxiter=0, etype=1, eps=1.0E-8, min_det=0.0, itype=3)
    
    # Use WORLD vocoder to spectral envelope
    _, sp2, _ = pyworld.wav2world(wav2.astype(np.double), fs=sr,frame_period=FRAME_PERIOD, fft_size=fft_size)
    # Extract MCEP features
    mgc2 = pysptk.sptk.mcep(sp2, order=mcep_dim, alpha=alpha, maxiter=0, etype=1, eps=1.0E-8, min_det=0.0, itype=3)

    result = melcd(mgc1[:,1:], mgc2[:,1:], lengths = None)
    return result

def mel2snd(mel_recon,mel_real,pad, HifiG):
        
    recon = torch.FloatTensor(mel_recon[:,0:pad])
    real = torch.FloatTensor(mel_real[:,0:pad])
    snd_recon = HifiG(recon.unsqueeze(0)).detach().numpy().squeeze()
    snd_real = HifiG(real.unsqueeze(0)).detach().numpy().squeeze()
    mcd = cal_mcd(snd_recon,snd_real)

    return snd_recon, snd_real, mcd

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None

    return sorted(cp_list)[-1]


class Dataset_mel(Dataset):
    """ Diabetes dataset."""

    # Initialize data, download, etc.
    def __init__(self, x, y, src_mask,pad_idx,nbin=128):
        self.len = x.shape[0]
        self.x_data_enc = torch.FloatTensor(x)
        self.y_data = torch.FloatTensor(y)
        self.src_mask = torch.IntTensor(src_mask)
        self.pad_idx = pad_idx
        self.nbin = nbin

    def __getitem__(self, index):
        max_start = max(0,self.pad_idx[index] - self.nbin)
        mel_start = random.randint(0,max_start)
        return self.x_data_enc[index,mel_start:mel_start+self.nbin],  self.y_data[index,mel_start:mel_start+self.nbin], self.src_mask[index,:,mel_start:mel_start+self.nbin], self.pad_idx[index]

    def __len__(self):
        return self.len


import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.first = True
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_last_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_last_lr()):
            param_group['lr'] = lr

