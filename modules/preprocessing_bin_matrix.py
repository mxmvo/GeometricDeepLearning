import numpy as np
import sys, scipy, os, re
sys.path.append('..')

from modules.trimesh import trimesh
from modules.fast_marching_method import FMM
from modules.gradient_walk.linear_walk import LinearWalk
from modules.adj_matrix import adjacency_matrix_fmm, adjacency_matrix_heat
#from modules.ddg import discrete_gradient
#from modules.geometry_functions import discrete_gradient
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from glob import glob
import torch

def read_ply(f_name):
    # Read the vertices and triangles from a ply file
    data = PlyData.read(f_name)
    
    data_vert = np.vstack([list(vertex) for vertex in data['vertex'].data])
    data_tri = np.vstack(data['face'].data['vertex_indices'])
    return trimesh(data_vert, data_tri)

data_path_in = '/home/maxim/dataset/registrations/'
data_path_out = '/home/maxim/dataset/alligned/'

start_file = int(sys.argv[1])
stop_file = int(sys.argv[2])

registrations = sorted(glob(os.path.join(data_path_in,'*.ply')))

for file_path in registrations[start_file:stop_file]:
    file = re.findall('tr_.*?ply$', file_path)[0]
    
    file_out = re.sub('\.ply$','_all.npz',file)
    
    path_out = os.path.join(data_path_out, file_out)
    
    print('-'*20)
    print('Starting:',file)
    if os.path.isfile(path_out):
        print('File {} already exists, continuing with next'.format(file))
        continue
    
    mesh = read_ply(file_path)
    adj_mat = adjacency_matrix_fmm(mesh,p_max = 0.03)

    print('Saving sparse matrix', end = ' : ')
    num_vert = len(mesh.vertices)
    scipy.sparse.save_npz(path_out, scipy.sparse.csc_matrix(adj_mat.reshape(num_vert,-1))) 
    print('Succesful')

    del mesh, adj_mat
