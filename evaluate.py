import sys, glob, os, re, pickle

from plyfile import PlyData, PlyElement

from tqdm import tqdm

from modules.trimesh import trimesh
from modules.geometry_functions import geometry_functions, laplace_beltrami_eigenfunctions

import time
import scipy
import scipy.sparse as sparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from modules.training.models import GCCN_6, GCCN_5, GCCN_4, GCCN_3, GCCN_2
from modules.training.loaders import BodyDataset
from modules.training.train import training
from modules.training.loss import siamese_loss

g_files = sorted(glob.glob('../dataset/g_func/*.pt'))[-20:]
c_files = sorted(glob.glob('../dataset/alligned_adj/*'))[-20:]
d_files = sorted(glob.glob('../dataset/dist_points_fmm/*points.npy'))[-20:]
p_files = sorted(glob.glob('../dataset/good_points/*.npz'))[-20:]

def main():
    #imodel_paths = ['../models/inv_64_32_16_run_','../models/inv_64_64_16_run_','../models/equi_64_8_16_run_','../models/equi_64_16_16_run_','../models/equi6_32_16_16_run_','../models/equi6_32_32_16_run_']
    model_paths = ['../models/big_inv_64_64_16_run_','../models/big_equi6_32_32_16_run_']
    neurons = [[150,64,64,16],[150,32,32,16]]
    types = [GCCN_3,GCCN_6]
    
    ind = int(sys.argv[1])
    
    m_subset = False

    for run in range(1,2):
        m_path = model_paths[ind]+str(run)+'/descr_10499.mdl'
        m_type = types[ind]
        m_neurons = neurons[ind]
        
        print('-'*50)
        print('Starting with: ', m_path)
        
        out_path = re.sub('\.mdl','.pickle', re.sub('/','_',m_path.lstrip('./')))
	
        if m_subset:
            out_path = os.path.join('./validation_results_subset_no_head',out_path)
        else:
            out_path = os.path.join('./validation_results',out_path)
	
        print(out_path)
	
        with open(m_path, 'rb') as f:
            saved_model = torch.load(f, map_location = 'cpu')
	
        model = m_type(m_neurons)
        model.load_state_dict(saved_model[0])
	
        big_mat = big_mat_calc(model)
        print('Evaluating')
	    
        if m_subset:
            sub = np.load('dense_points_no_head.npy')
            sublist = np.where(sub == True)[0]
        else:
            sublist = np.arange(6890)
	
        save_arr = evaluate(big_mat , sublist = sublist)
	
        print('Saving')
	
        with open(out_path,'wb') as f:
            pickle.dump(save_arr, f)


def evaluate(big_mat, sublist, p_files = p_files, d_files = d_files):
    subset = set(sublist)

    g_plot_x = np.linspace(0,1,num=200)
    g_plot_y = np.zeros_like(g_plot_x)

    cmc_x = [1,2,3,4,5,6,7,8,9,10,15,20,30,40,60,80,100]
    cmc_y = np.zeros_like(cmc_x, dtype = float)

    roc_thres = np.linspace(0,1.5,num = 100)**2
    roc_x = np.zeros_like(roc_thres)
    roc_y = np.zeros_like(roc_thres)

    TP = 0
    FP = 0
    for j in ([1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19]):
        g_dist_mat = np.load(d_files[j])
        point_mat = sparse.load_npz(p_files[j]).todense()
        d = (j //10)*10
        for i in tqdm(subset):
            # calculate the distance in the embedded space
            orig = big_mat[d,i]
            dist = np.linalg.norm(big_mat[j,sublist] - orig, axis = -1)


            good_points = set(np.where(point_mat[i] == 1)[1])
            good_points = good_points&subset
            TP += len(good_points)
            FP += len(subset)-len(good_points)

            # Geodesic
            # Find the closest neighbor index
            min_ind = np.argmin(dist)
            min_ind = sublist[min_ind]

            geo_dist = g_dist_mat[i,min_ind]
            g_plot_y[g_plot_x > geo_dist] += 1

            # CMC
            # Idea change argpartition and do argsort
            for x, k in enumerate(cmc_x):
                pif = np.argpartition(dist, k)[:k]
                pif = set(sublist[pif])
                if len(good_points & pif) > 0:
                    cmc_y[x]+= 1

            #ROC
            plot_values = np.zeros((len(roc_thres),2))
            #ind = 0
            #count = {i:0 for i in range(10)}

            for t, thres in enumerate(roc_thres):
                pif = np.where(dist < thres)[0]
                pif = set(sublist[pif])
                roc_y[t] += len(good_points & pif)
                roc_x[t] += len(pif - good_points)


    roc_y /= TP     # Devide by TP
    roc_x /= FP     # Devide by FP

    cmc_y /= len(subset)*18     # total matches made

    g_plot_y /= len(subset)*18  # Total matches made

    return [big_mat, [cmc_x, cmc_y],[roc_x, roc_y],[g_plot_x, g_plot_y]]

def big_mat_calc(model, c_files = c_files, g_files = g_files):
    res = {}

    int_res = {}
    for i in range(len(g_files)):
        g = torch.load(g_files[i])
        c_dict = torch.load(c_files[i])
        c = torch.sparse.FloatTensor(c_dict['ind'].to(torch.long), c_dict['data'].to(torch.float), torch.Size(c_dict['size']))

        with torch.no_grad():
            int_res[i] = model(g, c).data.numpy()
    return np.stack(int_res.values())


main()
