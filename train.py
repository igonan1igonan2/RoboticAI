import torch.nn as nn
import torch
import time
import numpy as np

"""
The architecture is based on the paper “Attention Is All You Need”. 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.
"""
import scipy.io as sio
from utils import scan_checkpoint, mel2snd
import os
import pandas as pd

def train(dataloader, model, optimizer, scheduler, args, train_data, test_data, g_hifi, subnum, cv, stt):
    start = 0
    min_val_loss = float('inf')
    tr_loss = {}
    tr_loss['G'] = np.zeros(args.epoch)
    tr_loss['L2'] = np.zeros(args.epoch)
    if not os.path.exists(args.path_to_save_folder):
        os.mkdir(args.path_to_save_folder)

    if os.path.isdir(args.path_to_save_model):
        cp_g = scan_checkpoint(args.path_to_save_model, 'G_')
        print(cp_g)
        if cp_g is not None:
            G = torch.load(cp_g, map_location='cpu')
            model['G'].load_state_dict(G['model'])
            optimizer['G'].load_state_dict(G['optim'])
            if scheduler['G'] is not None:
                scheduler['G'].load_state_dict(G['sche'])

            start = G['epoch']
            min_val_loss = G['min_val_loss']
    else:
        os.makedirs(args.path_to_save_model)
    if not os.path.isdir(args.path_to_save_mat):
        os.makedirs(args.path_to_save_mat)
    arg = {}
    arg['args'] = args
    sio.savemat(args.path_to_save_mat + '/args.mat', arg)

    model['G'].to(args.device)
    for state in optimizer['G'].state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(args.device)
    ##if not start==0:
    model['G'].eval()
    save_result(model, g_hifi, train_data, test_data, start+1, args,subnum,cv,stt,tr_loss)
    model['G'].train()

    print("Start to train classifier")
    lr_g = []
    model['G'].train()
    for epo in range(start, args.epoch):
        if scheduler['G'] is not None:
            lr_g += scheduler['G'].get_last_lr()
        start_time = time.time()
        train_loss = 0
        train_mse_loss = 0
        for src, answ, src_mask, _ in dataloader:
            src = src.to(args.device)
            src_mask = src_mask.to(args.device)
            answ = answ.to(args.device)

            optimizer['G'].zero_grad()
            prediction = model['G'](src, src_mask)

            del src, src_mask
            torch.cuda.empty_cache()

            mse_loss = nn.MSELoss()(prediction, answ)

            del prediction, answ
            torch.cuda.empty_cache()

            loss_g = mse_loss * args.mseloss
            loss_g.backward()
            optimizer['G'].step()

            train_mse_loss += mse_loss.cpu().detach().item()
            train_loss += loss_g.cpu().detach().item()
            del loss_g, mse_loss
            torch.cuda.empty_cache()

        tr_loss['G'][epo] = train_loss
        tr_loss['L2'][epo] = train_mse_loss

        # print(epo+1)
        elapsed = time.time() - start_time

        print('Sub {:3d} | CV {:3d} | epoch {:3d} | {:5d} batches | '
              '{:5.2f} s | scheduler_lr {:5.5f} | optimizer_lr {:5.5f} | '
              ' train_mse_loss {:5.5f}| train_g_loss {:5.5f} |'.format(
            subnum, cv, epo + 1, len(dataloader), elapsed, scheduler['G'].get_last_lr()[0],
            optimizer['G'].param_groups[0]['lr'],
                        train_mse_loss / len(dataloader),
                        train_loss / len(dataloader)))

        if (torch.IntTensor(args.save_epochs) == (epo + 1)).sum():
            G_dict = {'model': model['G'].state_dict(),
                      'optim': optimizer['G'].state_dict(),
                      'min_val_loss': min_val_loss,
                      'epoch': epo + 1
                      }
            if scheduler['G'] is not None:
                G_dict['sche'] = scheduler['G'].state_dict()
            print("Start Saving Model")
            torch.save(G_dict, args.path_to_save_model + 'G_{:04d}'.format(epo + 1))

        if (torch.IntTensor(args.print_epochs) == (epo + 1)).sum():
            model['G'].eval()
            save_result(model, g_hifi, train_data, test_data, epo + 1, args, subnum, cv, stt, tr_loss)
            model['G'].train()

        scheduler['G'].step(epo)


def save_result(model, g_hifi, train_data, test_data, epo, args, sub, cv, stt, train_loss, train=True):
    print("Start Saving Result")

    indices = sio.loadmat(args.cv_ind)['cv_ind'][0][sub].squeeze()
    idx = sio.loadmat('trig/sub' + str(sub) + '/trig.mat')['trig_idx'].squeeze()
    cv_ind = idx[indices == cv]
    val_loss = 0
    val_mse_loss = 0

    te_answ = []
    te_recon = []

    te_pad = []
    with torch.no_grad():
        for src, answ, src_mask, pad_idx in test_data:
            prediction = model['G'](src.to(args.device), src_mask.to(args.device))
            te_answ += answ
            te_recon += prediction.to('cpu').detach()
            te_pad += pad_idx

            val_mse_loss += nn.MSELoss()(prediction, answ.to(args.device)).cpu().detach().item()
            del prediction, src, src_mask, answ
            torch.cuda.empty_cache()

        val_loss += ( val_mse_loss * args.mseloss)

        torch.cuda.empty_cache()

        te_answ = np.array(torch.stack(te_answ).cpu().detach().numpy()).transpose(0, 2, 1)
        te_recon = np.array(torch.stack(te_recon).cpu().detach().numpy()).transpose(0, 2, 1)

        te_pad = np.array(torch.stack(te_pad).cpu().detach().numpy())

    test = {}

    test['g_loss'] = train_loss['G']

    test['l2_loss'] = train_loss['L2']

    test['te_real'] = te_answ
    test['te_recon'] = te_recon
    test['te_pad'] = te_pad
    test['cv_ind'] = {}
    test['snd_test'] = {}
    test['mcd_test'] = {}
    test['mel_test'] = {}
    mcd_test = np.zeros(te_recon.shape[0])


    ind_test = np.zeros(te_recon.shape[0])

    for test_idx in range(te_recon.shape[0]):
        snd_recon, snd_real, mcd_test[test_idx]= mel2snd(te_recon[test_idx, :, :], te_answ[test_idx, :, :],
                                te_pad[test_idx,].squeeze(), g_hifi)


        ind_test[test_idx] = cv_ind[test_idx]
        test['snd_test']['recon' + str(test_idx + 1)] = snd_recon
        test['snd_test']['real' + str(test_idx + 1)] = snd_real
        test['cv_ind'] = ind_test
        test['mcd_test'] = mcd_test
        test['mel_test']['recon' + str(test_idx + 1)] = te_recon[test_idx, :, 0:te_pad[test_idx,].squeeze()]
        test['mel_test']['real' + str(test_idx + 1)] = te_answ[test_idx, :, 0:te_pad[test_idx,].squeeze()]

        torch.cuda.empty_cache()
    test['epo'] = epo + 1
    test['val_loss'] = val_loss / len(test_data)
    if train:
        sio.savemat(args.path_to_save_mat + 'test' + str(epo) + '.mat', test)
    else:
        sio.savemat(args.path_to_save_mat + 'test' + str(epo) + '_.mat', test)
    del test
    del te_recon, te_answ, mcd_test
    torch.cuda.empty_cache()