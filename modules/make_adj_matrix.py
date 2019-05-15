import numpy as np
import sys, scipy, os, re
sys.path.append('..')

from modules.trimesh import trimesh
from modules.fast_marching_method import FMM
from modules.gradient_walk.linear_walk import LinearWalk
#from modules.ddg import discrete_gradient
#from modules.geometry_functions import discrete_gradient
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from glob import glob

def trimesh_from_ply(f_name):
    data = PlyData.read(f_name)
    
    data_vert = np.vstack([list(vertex) for vertex in data['vertex'].data])
    data_tri = np.vstack(data['face'].data['vertex_indices'])
    return trimesh(data_vert,data_tri)

data_path_in = '/Users/maximoldenbeek/Dropbox/Thesis/datasets/MPI-FAUST/training/registrations'
data_path_out = '/Users/maximoldenbeek/Dropbox/Thesis/datasets/FAUST_preprocesed/raw'

registrations = sorted(glob(os.path.join(data_path_in,'*.ply')))

for file_path in registrations:
    file = re.findall('tr_.*?ply$', file_path)[0]
    file_out = re.sub('\.ply$','_raw.npz',file)
    path_out = os.path.join(data_path_out, file_out)
    print('-'*20)
    print('Starting:',file)
    if os.path.isfile(path_out):
        print('File {} already exists, continuing with next'.format(file))
        continue
    
    mesh = trimesh_from_ply(file_path)
    mat = adjacency_matrix_fmm(mesh,p_max = 0.03)

    print('Saving sparse matrix')
    num_vert = len(mesh.vertices)
    scipy.sparse.save_npz(path_out, scipy.sparse.csc_matrix(mat.reshape(num_vert,-1))) 
    print('Succesfull')