import re, os
import sys
sys.path.append('..')

import torch
import numpy as np
from modules.geometry_functions import geometry_functions
from modules.trimesh import trimesh
from plyfile import PlyData, PlyElement
from glob import glob
from tqdm import tqdm

def read_ply(f_name):
    # Read the vertices and triangles from a ply file
    data = PlyData.read(f_name)
    
    data_vert = np.vstack([list(vertex) for vertex in data['vertex'].data])
    data_tri = np.vstack(data['face'].data['vertex_indices'])
    return trimesh(data_vert, data_tri)


registrations = glob('/Users/maximoldenbeek/Dropbox/Thesis/datasets/MPI-FAUST/training/registrations/*ply')

dir_out = '/Users/maximoldenbeek/Dropbox/Thesis/datasets/FAUST_preprocesed/geometry_functions_norm_norm/'

for file_mesh in tqdm(registrations):
    f_base = re.findall(r'(tr_reg_[0-9]+)', file_mesh)[0]
    file_out = os.path.join(dir_out, f_base+'_g_func.pt')
    mesh = read_ply(file_mesh)
    g_func = geometry_functions(mesh)

    with open(file_out, 'wb') as f:
        torch.save(torch.Tensor(g_func.T),f)

    # Save G_func as tensor
