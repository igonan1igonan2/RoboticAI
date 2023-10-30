import argparse

from torch.utils.data import DataLoader

from model_conformer import Conformer_Mel
from train import *
from utils import *

import numpy as np
import random
import os
import json
from models_hifi import Generator
# import whisper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['KMP_DUPLICATE_LIB_OK']='True'

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(args, subnum, CV):
    args.save_folder = 'result_' + args.mel_style

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    args.path_to_save_model = args.save_folder + '/save_model/sub' + str(subnum) + '/CV' + str(CV) + '/'
    args.path_to_save_mat = args.save_folder + '/save_mat/sub' + str(subnum) + '/CV' + str(CV) + '/'

    temp_sig = sio.loadmat(args.path_to_data_folder + '/data/feat/sub' + str(subnum) + '/sig_flip_padd.mat')
    temp_sound = sio.loadmat(args.path_to_data_folder + '/data/feat/sub' + str(subnum) + '/mels_' + args.mel_style + '_flip_padd.mat')
    max_len = int(temp_sig['max_len'].item())
    args.max_len = max_len
    src_mask = temp_sig['sig_mask'][:, np.newaxis, :]
    sig = temp_sig['sig_pad']
    sound = temp_sound['mels_pad']
    pad_idx = temp_sig['pad_idx'].astype('int32')

    temp_idx = sio.loadmat(args.cv_ind)['cv_ind'].squeeze()
    idx = temp_idx[subnum].squeeze()

    tr_idx = idx != CV
    te_idx = idx == CV

    tr_sig = sig[tr_idx,:,0:args.nCh]
    tr_src_mask = src_mask[tr_idx,]
    tr_snd = sound[tr_idx,]
    tr_pad = pad_idx[tr_idx,]

    te_sig = sig[te_idx,:,0:args.nCh]
    te_src_mask = src_mask[te_idx,]
    te_snd = sound[te_idx,]
    te_pad = pad_idx[te_idx,]


    tr_dataset = Dataset_mel(x=tr_sig, y=tr_snd, src_mask=tr_src_mask, pad_idx=tr_pad, nbin=args.nBin)
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)

    te_dataset = Dataset_mel(x=te_sig, y=te_snd, src_mask=te_src_mask, pad_idx=te_pad, nbin=max_len)
    te_dataloader = DataLoader(te_dataset, batch_size=1, shuffle=False)

    tr_val_dataset = Dataset_mel(x=tr_sig, y=tr_snd, src_mask=tr_src_mask, pad_idx=tr_pad, nbin=max_len)
    tr_val_dataloader = DataLoader(tr_val_dataset, batch_size=1, shuffle=False)

    model = {}
    model['G'] = Conformer_Mel(d_input=args.nCh, d_output=args.nMel, d_model=args.d_model, nhead=args.nhead,
                                  e_layers=args.e_layers, d_layers=args.d_layers, d_bi=args.d_bi,
                                  ff_dropout=args.ff_dropout, ff_expansion_factor=args.ff_ex_factor,
                                  conv_expansion_factor=args.conv_ex_factor,
                                  conv_dropout=args.conv_dropout).to(args.device)

    optimizer = {}
    optimizer['G'] = torch.optim.AdamW(model['G'].parameters(), lr=args.lr)
    scheduler = {}
    scheduler['G'] = CosineAnnealingWarmUpRestarts(optimizer['G'], first_cycle_steps=args.first_cycle_steps,
                                                   cycle_mult=args.cycle_mult, max_lr=args.max_lr, min_lr=args.min_lr,
                                                   warmup_steps=args.warmup, gamma=args.gamma)

    with open(args.config_name) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    state_dic_g = load_checkpoint(args.g_weight_name, 'cpu')
    generator = Generator(h)
    generator.load_state_dict(state_dic_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    print('Sub = ' + str(subnum) + ' | ' + 'CV = ' + str(CV))

    train(tr_dataloader, model, optimizer, scheduler, args, tr_val_dataloader, te_dataloader, generator, subnum, CV)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_num", type=int, default=1)  ##GNWS
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--cv_ind", type=str, default='cv_ind_615.mat')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_sub", type=str, default="/")
    parser.add_argument("--path_to_data_folder", type=str, default=os.getcwd())
    parser.add_argument("--path_to_save_folder", type=str, default="/")
    parser.add_argument("--path_to_save_model", type=str, default="/save_model/")
    parser.add_argument("--path_to_save_loss", type=str, default="/save_loss/")

    parser.add_argument("--path_to_save_mat", type=str, default="/save_mat/")
    parser.add_argument("--mel_style", type=str, default="nr")
    parser.add_argument("--print_every", type=int, default=1000)
    parser.add_argument("--print_epochs", type=int, default=[500,1000])
    parser.add_argument("--save_epochs", type=int, default=[1000])

    parser.add_argument("--nMel", type=int, default=80)
    parser.add_argument("--nCh", type=int, default=15)
    parser.add_argument("--d_model", type=int, default=2048)
    parser.add_argument("--nhead", type=int, default=32)
    parser.add_argument("--e_layers", type=int, default=4)
    parser.add_argument("--d_layers", type=int, default=2)
    parser.add_argument("--d_bi", type=bool, default=True)
    parser.add_argument("--kernel_size", type=int, default=31)
    parser.add_argument("--mseloss", type=int, default=1)

    parser.add_argument("--nBin", type=int, default=128)
    parser.add_argument("--conv_dropout", type=float, default=0.1)
    parser.add_argument("--ff_dropout", type=float, default=0.1)
    parser.add_argument("--ff_ex_factor", type=float, default=3)
    parser.add_argument("--conv_ex_factor", type=float, default=2)

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_decay", type=float, default=0.999)

    parser.add_argument("--first_cycle_steps", type=int, default=500)
    parser.add_argument("--cycle_mult", type=int, default=1)
    parser.add_argument("--max_lr", type=float, default=0.0001)
    parser.add_argument("--min_lr", type=float, default=0.00001)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.9)

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--config_name", type=str, default="config.json")
    parser.add_argument("--g_weight_name", type=str, default="g_1000")

    args = parser.parse_args()

    subnum = 0
    # sttmodel = whisper.load_model('large').to('cpu')
    sub=[1]  #[

    CVs = [1]

    for cv in CVs:
        for subnum in sub:
            main(
                args=args, subnum=subnum, CV=cv
            )