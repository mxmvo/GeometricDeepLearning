import sys, glob, os, re

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def write_log(line, file):
    with open(file,'a') as f:
        f.write(line)

def training(model, dataloader, params, batch_loss):
    opt = torch.optim.Adam(model.parameters(), lr = params['lr'])
    vert_list = list(range(params['n_vert']))
    batch_size = params['batch_size']
    len_data_set = len(dataloader)
    
    log_file = os.path.join(params['model_dir'], 'log.txt')
    print(params)
    t_start = t_last = time.time()
    for ep in range(params['epochs']):
        avg_loss = []
        for i, batch in enumerate(dataloader):
            inp_1 , inp_2 = batch

            g_1, g_2 = inp_1[0][0], inp_2[0][0]
            d_1, d_2 = inp_1[1], inp_2[1]

            c_1 = torch.sparse.FloatTensor(d_1['ind'][0], d_1['data'][0], torch.Size(d_1['size']))
            c_2 = torch.sparse.FloatTensor(d_2['ind'][0], d_2['data'][0], torch.Size(d_2['size']))

            # To device
            g_1, g_2 = g_1.to(params['device']), g_2.to(params['device'])
            c_1, c_2 = c_1.to(params['device']), c_2.to(params['device'])

            out_1 = model(g_1, c_1)
            out_2 = model(g_2, c_2)

            ind = np.random.choice(vert_list, size = 2*batch_size, replace = False)

            out = out_1[ind[:batch_size]]
            out_pos = out_2[ind[:batch_size]]
            out_neg = out_2[ind[batch_size:]]

            loss = batch_loss(out,out_pos, out_neg, params)

            loss.backward()
            print('\r{:>10}: {:.5f}'.format(ep*len_data_set + i,loss), end = '')
            avg_loss.append(loss.data.numpy())
            opt.step()
            opt.zero_grad()

            if (i % params['it_print']) == 0:
                t_new = time.time()
                line ='\n Tot time :{:.2f} min, Iter time: {:.2f} sec, avg loss: {}'.format((t_new-t_start)/60, t_new-t_last, np.mean(avg_loss)) 
                print(line)
                write_log(line, log_file)
                t_last = t_new
                avg_loss = []

            if ((params['it']+ep*len_data_set+i) % params['it_save']) == 0:
                m_file = os.path.join(params['model_dir'], 'descr_'+str(params['it']+ep*len_data_set+i)+'.mdl')
                log_line = 'Saving Model: {}'.format(m_file)
                print(log_line, end ='...', flush = True)
                torch.save(model.state_dict(), m_file)
                print('Saved', flush = True)
                write_log(log_line, log_file)

