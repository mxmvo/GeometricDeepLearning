import numpy as np
import sys, scipy, os, re
sys.path.append('..')

from modules.trimesh import trimesh
from modules.fast_marching_method import FMM
from modules.gradient_walk.linear_walk import LinearWalk
from modules.geometry_functions import discrete_gradient
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from glob import glob

def trimesh_from_ply(f_name):
    data = PlyData.read(f_name)
    
    data_vert = np.vstack([list(vertex) for vertex in data['vertex'].data])
    data_tri = np.vstack(data['face'].data['vertex_indices'])
    return trimesh(data_vert,data_tri)

def adjacency_matrix_fmm(mesh, p_max =.05,  p_bins = 5, t_bins = 16, range_ind = None):
    #t0 = time.time()
    num_vert = len(mesh.vertices)

    if range_ind == None:
        range_ind = range(num_vert)


    #paths = {}
    
    fmm = FMM(mesh)
    
    row_coo = []
    col_coo = []
    angle_coo = []
    radius_coo = []
    
    times = []
    for step, ind in enumerate(tqdm(range_ind)):
        
        phi = fmm.run([ind], max_distance = 2*p_max)
        phi[phi == np.inf] = 3*p_max
        
        # Propegating the points backwards
        # Find the points that we are interested in
        interior = np.where(phi < p_max)[0]
       

        walk = LinearWalk(mesh)
        ext_paths = []
        Y = discrete_gradient(mesh, phi)
        for p in (set(interior)-{ind}):
            ext_paths.append(walk.run(ind, p, -Y, max_length = 2*p_max))
        
        
        
        # Calculate the point of entry in the 1-ring
        nbh = mesh.chart(ind)['nbh']
        
        pnt_2d = []
        angle = []    
        indices = []
        stop_faces = [set(t) for t in mesh.triangles_containing_node(ind)]

        for path in ext_paths:
            if path['faces'][-1] not in stop_faces:
                continue
            indices.append(path['start_ind'])
            
            pnt = mesh.mesh_to_chart(path['points'][-1], path['faces'][-1], nbh)
            pnt_2d.append( pnt/(np.linalg.norm(pnt)))
            
        
        pnt_2d = np.array(pnt_2d)
        if len(pnt_2d)>0:
            angle = np.arccos(np.clip(pnt_2d[:,0],-1,1))
            angle[pnt_2d[:,1]< 0] = 2*np.pi - angle[pnt_2d[:,1] < 0]

            angle_dict = dict(zip(indices, angle))


            for v_ind in angle_dict.keys():
                t = min(int((angle_dict[v_ind])/(2*np.pi)*t_bins), t_bins - 1)
                r = min(int((phi[v_ind]/p_max)*p_bins), p_bins - 1)
                
                row_coo.append(ind)
                col_coo.append(v_ind)
                angle_coo.append(t)
                radius_coo.append(r)
                
    adj_mat = np.zeros((num_vert, num_vert,p_bins,t_bins), dtype = np.int8)
    adj_mat[row_coo,col_coo,radius_coo,angle_coo] = 1

    return adj_mat 
    #return scipy.sparse.csc_matrix(adj_mat.reshape(num_vert,-1))
 

data_path_in = '/Users/maximoldenbeek/Dropbox/Thesis/datasets/MPI-FAUST/training/registrations'
data_path_out = '/Users/maximoldenbeek/Dropbox/Thesis/datasets/FAUST_preprocesed/hard'

registrations = sorted(glob(os.path.join(data_path_in,'*.ply')))

for file_path in registrations:
    file = re.findall('tr_.*?ply$', file_path)[0]
    file_out = re.sub('\.ply$','_hard.npz',file)
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