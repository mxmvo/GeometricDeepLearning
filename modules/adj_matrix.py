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


def adjacency_matrix_fmm(mesh, p_max =.05,  p_bins = 5, t_bins = 16, range_ind = None):
    
    num_vert = len(mesh.vertices)

    if range_ind == None:
        range_ind = range(num_vert)

    # Use the Fast Marching Method to calculate distances
    fmm = FMM(mesh)
    
    # Keep track of indices
    row_coo = []
    col_coo = []
    angle_coo = []
    radius_coo = []
    
    for ind in tqdm(range_ind):

        # Calculate the distances.
        phi = fmm.run([ind], max_distance = 2*p_max)
        phi[phi == np.inf] = 3*p_max
        
        # Propegating the points backwards
        # Find the points that we are interested in
        interior = np.where(phi < p_max)[0]
       
        # Walk that walk
        walk = LinearWalk(mesh)
        ext_paths = []
        
        for p in (set(interior)-{ind}):
            ext_paths.append(walk.run(ind, p, phi, max_length = 2*p_max))
        
        
        
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
            
        # Add the origin
        row_coo.append(ind)
        col_coo.append(ind)
        angle_coo.append(0)
        radius_coo.append(0)

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
 