import numpy as np
import sys, scipy, os, re
sys.path.append('..')


from scipy import sparse
from scipy.sparse.linalg import spsolve
from modules.trimesh import trimesh
from modules.fast_marching_method import FMM
from modules.gradient_walk.linear_walk import LinearWalk
from modules.ddg import discrete_connection_laplacian, mean_dist, discrete_gradient
from modules.ddg import discrete_laplacian, make_rotation_matrix, mean_dist, max_dist
from modules.heat_method import heat_method



#from modules.geometry_functions import discrete_gradient
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from glob import glob

from scipy.sparse.linalg import spsolve


def adjacency_matrix_fmm(mesh, p_max =.05,  p_bins = 5, t_bins = 16, range_ind = None):
    
    num_vert = len(mesh.vertices)

    if range_ind == None:
        range_ind = range(num_vert)

    # Use the Fast Marching Method to calculate distances
    fmm = FMM(mesh)
    

    conn_laplace, area_mat = discrete_connection_laplacian(mesh)
    h2 = mean_dist(mesh.vertices[mesh.triangles])
    X = np.zeros((num_vert, 1), dtype = np.csingle)
    X[0] = 1+0j
    angle_field = spsolve((area_mat-h2*conn_laplace), X)
    angle_field = np.angle(angle_field) 
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
            
            #pnt_2d.append( pnt/(np.linalg.norm(pnt)))
            angle.append(np.angle(pnt[0]+pnt[1]*1j))
            
        # Add the origin
        row_coo.append(ind)
        col_coo.append(ind)
        angle_coo.append(0)
        radius_coo.append(0)

        pnt_2d = np.array(pnt_2d)
        #if len(pnt_2d)>0:
        if len(angle) > 0:
            # Correct here with the calculated vector field
            # Change code to make use of complex numbers
            # Maybe not sparse? Percentage wise this is not the best optimization

            #angle = np.arccos(np.clip(pnt_2d[:,0],-1,1))
            #angle[pnt_2d[:,1]< 0] = 2*np.pi - angle[pnt_2d[:,1] < 0]

            angle_dict = dict(zip(indices, angle))


            for v_ind in angle_dict.keys():
                a = (angle_dict[v_ind]-angle_field[ind]) % (np.pi*2)
                t = min(int(a/(2*np.pi)*t_bins), t_bins - 1)
                r = min(int((phi[v_ind]/p_max)*p_bins), p_bins - 1)
                
                row_coo.append(ind)
                col_coo.append(v_ind)
                angle_coo.append(t)
                radius_coo.append(r)
                
    adj_mat = np.zeros((num_vert, num_vert,p_bins,t_bins), dtype = np.int8)
    adj_mat[row_coo,col_coo,radius_coo,angle_coo] = 1

    return adj_mat 

def adjacency_matrix_heat(mesh, p_max =.05,  p_bins = 5, t_bins = 16, range_ind = None):
    
    num_vert = len(mesh.vertices)
 
    
    rot_mat = make_rotation_matrix(mesh)
    rot_mat_csr = rot_mat.tocsr()
    cot_mat, area_mat = discrete_laplacian(mesh)

    disc_conn_laplace = rot_mat.multiply(cot_mat)
    h_mean = mean_dist(mesh.vertices[mesh.triangles])

    mat_con = sparse.linalg.factorized(area_mat - h_mean*disc_conn_laplace) 


    heat = heat_method(mesh.vertices, mesh.triangles)
    # Calculate horizontal
    X = np.zeros((num_vert, 1), dtype = np.csingle)
    X[0] = 1+0j
    X_hor = mat_con(X)

    bin_mat = np.zeros((num_vert, num_vert,p_bins,t_bins), dtype = np.int8)
    for v in tqdm(range(num_vert)):

        phi, _ = heat.run(v)

        hor_angle = X_hor[v]
        X = np.zeros((num_vert,1), dtype = np.csingle)
        X[v] = hor_angle/np.abs(hor_angle)
        X_hor = mat_con(X)    
        
        x_tilde = np.zeros(len(mesh.vertices), dtype = np.csingle)
        ind = mesh.chart(v)['sort_ind']
        angles = mesh.chart(v)['angles']
        for i in range(len(ind)-1):
            x_tilde[ind[i]] = -np.exp(1j*angles[i])
        X_rad = np.multiply(np.conj(rot_mat_csr[v].todense()),(x_tilde)).reshape(-1,1)
        X_rad = mat_con(X_rad)



        angle_field = np.array(np.angle(np.multiply(X_rad, np.conj(X_hor)))).reshape(-1) % (2*np.pi)
        dist_points = np.where(phi < p_max)[0]
        dist_values = phi[dist_points]
        dist_bins = np.abs(((dist_values/(p_max))*p_bins)//1).astype(int)
        angle_values = angle_field[dist_points]
        angle_bins = np.abs((angle_values/(2*np.pi)*t_bins)//1).astype(int)

        bin_mat[[v]*len(dist_points),dist_points, dist_bins, angle_bins] = 1    
   
    return bin_mat
 