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


# Some small functions
t_bins = 16
p_bins = 5

p_exp = lambda x,y: np.exp(-x**2/(2*y**2))
p_trans = lambda r: r/p_bins
t_trans = lambda t: 2*np.pi*t/t_bins

start_file = int(sys.argv[1])
stop_file = int(sys.argv[2])

# A function that returns the vertex areas
def vertex_area(mesh):
    tri = mesh.vertices[mesh.triangles]
    tri_area = np.linalg.norm(np.cross(tri[:,0]-tri[:,2], tri[:,1]-tri[:,2]), axis = 1)/6
    A_mat = np.zeros(len(mesh.vertices), dtype = np.float32)
    for i, t in enumerate(mesh.triangles):
        t1, t2, t3 = t
        A_mat[t1] += tri_area[i]
        A_mat[t2] += tri_area[i]
        A_mat[t3] += tri_area[i]
        
    return A_mat

def make_kernel(p_bins,t_bins):
    kernel = np.zeros((t_bins, p_bins))

    for i in range(p_bins):
        r_1 = p_trans(i)
        val = p_exp(r_1, .1)
        for j in range(t_bins):
            kernel[j,i] = val
    return kernel/kernel.sum()

kernel = make_kernel(p_bins, t_bins)

# A matrix containing the distance probabilities
def dist_mat(p_bins= 5, t_bins = 16):
    rho = list(range(p_bins))
    theta = list(range(t_bins))
    rho_o, theta_o = np.meshgrid(rho,theta)
    rho, theta = p_trans(rho_o), t_trans(theta_o)
    print(rho.shape)
    y1 = (rho*np.cos(theta)).reshape(-1)
    y2 = (rho*np.sin(theta)).reshape(-1)
    y = np.stack([y1,y2]).T
    dist = np.zeros((80,80), dtype = np.float32)

    p_exp = lambda x, y: np.exp(-x**2/(2*y**2))

    for i in range(80):
        for j in range(80):
            y1 = y[i]
            y2 = y[j]
            d = np.linalg.norm(y1 - y2)
            dist[i,j] = p_exp(d, .3)

    dist = dist/dist.sum(1, keepdims = True)
    return dist

dist = dist_mat()
def big_prob_mat(mesh, p_mat, dist = dist):
    N, _, R, T = adj_mat.shape
    B = T*R
    
    A_mat = vertex_area(mesh)

    p_mat = p_mat.transpose(0,1,3,2)
    p_mat = p_mat.reshape(N,N,B)
    
    # Insert the vertex areas
    # [N,N,B] hadamard [1,N,1]
    p_mat = np.multiply(p_mat, A_mat[np.newaxis,:,np.newaxis])
    
    # Insert the probabilities and normalize
    # [N,N,B] [N,B,B] => [N,N,B]
    p_mat = np.matmul(p_mat, dist)
    
    p_mat = p_mat/p_mat.sum(axis =1, keepdims = True)
    
    return p_mat.transpose(0,2,1)

def read_ply(f_name):
    # Read the vertices and triangles from a ply file
    data = PlyData.read(f_name)
    
    data_vert = np.vstack([list(vertex) for vertex in data['vertex'].data])
    data_tri = np.vstack(data['face'].data['vertex_indices'])
    return trimesh(data_vert, data_tri)

def read_adj(f_name):
    adj_mat = scipy.sparse.load_npz(f_name)
    N= adj_mat.shape[0]
    p_mat = adj_mat.toarray().reshape((N,N,5,16))
    return p_mat

data_path_reg = '/home/maxim/dataset/registrations'
data_path_bin = '/home/maxim/dataset/alligned'
data_path_out = '/home/maxim/dataset/alligned_adj'


bin_matrices = sorted(glob(os.path.join(data_path_bin,'*.npz')))[start_file:stop_file]
registrations = sorted(glob(os.path.join(data_path_reg,'*.ply')))[start_file:stop_file]

for i, file_path in enumerate(tqdm(bin_matrices)):
    mesh_path = registrations[i]
    
    file = re.findall('tr_.*?npz$', file_path)[0]
    

    file_out = re.sub('\.npz$','_conn.pt',file)
    
    path_out = os.path.join(data_path_out, file_out)
    
    print('-'*20)
    print('Starting:',file, mesh_path)
    if os.path.isfile(path_out):
        print('File {} already exists, continuing with next'.format(file))
        continue
    
    mesh = read_ply(mesh_path)
    adj_mat = read_adj(file_path)
    mat = big_prob_mat(mesh, adj_mat)
    mat = scipy.sparse.coo_matrix(mat.reshape(-1,6890))
    i = torch.LongTensor([mat.row,mat.col])
    v = torch.FloatTensor(mat.data)
    s = torch.Size(mat.shape)
    
    print('Saving smooth matrix', end = ' : ')
    torch.save({'ind':i, 'data':v, 'size':s}, path_out)
    print('Succesful')

    del mesh, mat, adj_mat
