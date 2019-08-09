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

from modules.training.models import GCCN_4, GCCN_3, GCCN_2
from modules.training.extra_layers import EquivariantLayer, GeodesicLayer, AMP
from modules.training.loaders import BodyDataset
from modules.training.train import training
from modules.training.loss import siamese_loss

# Training files
g_files = sorted(glob.glob('../dataset/g_func/*'))[:70]
c_files = sorted(glob.glob('../dataset/alligned_adj/*'))[:70]
params = {'batch_size':1500,
         'lr': 0.001,
         'epochs': 100,
         'model_dir': '/home/maxim/models/heat_equi_16_16_16_run_1/',
         'p_bins': 5,
         't_bins': 16,
         'n_vert': 6890,
         'it_print': 50,
         'it_save': 100,
         'it':None,
         'loss_mu':.2,
         'loss_gamma': .5,
         'optim':'Adam',
         'architecture': 'GCCN_3, equivariant relu',
         'neurons': [150,16,16,16],
         'device': torch.device("cuda:1")}

# Initialize Model
model = GCCN_4(params['neurons'], device = params['device'])


dataset = BodyDataset(g_files,c_files)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

summary = os.path.join(params['model_dir'], 'summary')
with open(summary, 'wb') as f:
    pickle.dump(params, f)

params = model.load_model(params)

model.to(params['device'])
training(model, dataloader, params, siamese_loss)

