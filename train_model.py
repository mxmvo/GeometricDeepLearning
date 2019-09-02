from scipy import sparse
import sys, glob, os, re, pickle


from modules.trimesh import trimesh
from modules.geometry_functions import geometry_functions

import time
import scipy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from modules.training.models import GCCN_6, GCCN_5, GCCN_4, GCCN_3, GCCN_2
from modules.training.extra_layers import EquivariantLayer, GeodesicLayer, AMP
from modules.training.loaders import BodyDataset
from modules.training.train import training
from modules.training.loss import siamese_loss

# Training files
g_files = sorted(glob.glob('../dataset/g_func/*'))[:70]
c_files = sorted(glob.glob('../dataset/alligned_adj/*'))[:70]
p_files = sorted(glob.glob('../dataset/good_points/*.npz'))[:70]

def train_model(params, g_files =g_files, c_files = c_files ,p_files= p_files):
    model = params['model'](params['neurons'], device = params['device'])
    
    if not os.path.isdir(params['model_dir']):
        os.makedirs(params['model_dir'])
        print('Made directory: ', params['model_dir'])
    else:
        print('Directory', params['model_dir'], 'already exists')

    if params['subset']:
        sub = np.load('dense_points.npy')
        sublist = np.where(sub ==  True)[0]
        dataset = BodyDataset(g_files,c_files, p_files, range_list = sublist, samples = params['batch_size'])
    else:
        dataset = BodyDataset(g_files,c_files, p_files, samples = params['batch_size'])

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    summary = os.path.join(params['model_dir'], 'summary')
    with open(summary, 'wb') as f:
        pickle.dump(params, f)

    params = model.load_model(params)

    model.to(params['device'])
    training(model, dataloader, params, siamese_loss)

params = {'batch_size':1000,
         'lr': 0.001,
         'epochs': 150,
         'p_bins': 5,
         't_bins': 16,
         'n_vert': 6890,
         'it_print':100,
         'it_save': 500,
         'it':None,
         'loss_mu':.2,
         'loss_gamma': .5,
         'optim':'Adam'}

params['subset'] = False
params['device'] = torch.device("cuda:1")

for i in range(2):
    params['model'] = GCCN_3
    params['model_dir'] = '/home/maxim/models/big_inv_64_64_16_run_'+str(i)+'/'
    params['neurons'] = [150,64,64,16]

    train_model(params)

    #params['model'] = GCCN_4
    #params['model_dir'] = '/home/maxim/models/sub_equi_64_16_16_run_'+str(i)+'/'
    #params['neurons'] = [150,64,16,16]

    #train_model(params)


    #params['model'] = GCCN_6
    #params['model_dir'] = '/home/maxim/models/big_equi6_32_32_16_run_'+str(i)+'/'
    #params['neurons'] = [150,32,32,16]

    #train_model(params)

