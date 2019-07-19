'''
This file is devoted to discrete geometry operators
'''

import numpy as np
from scipy import sparse

def calculate_cotangent(x):
    '''
    calculate the cotangent of the angle between the first column and the second column
    '''
    num = np.multiply(x[:,0],x[:,1]).sum(-1)
    den = np.linalg.norm(np.cross(x[:,0],x[:,1]), axis = -1)
    return num/(den)

def calculate_circumcenter(x):
    ''' 
    calculate the center of the circle inscribing the 2 vector in the first column and the origin
    
    given the vectors a and b this can be calculated as
        m = ((|a|^2 b- |b|^2a) x (a x b))/(2|a x b|^2)
    x means the cross product
    '''
    a, b= x[:,0], x[:,1]
    cross_ab = np.cross(a,b)
    
    num = b*np.linalg.norm(a, axis = 1, keepdims = True)**2- a*np.linalg.norm(b, axis = 1, keepdims = True)**2
    num = np.cross(num, cross_ab)
    den = 2*np.linalg.norm(cross_ab, axis = 1, keepdims = True)**2
    return num/(den)

def mean_dist(tri):
    tri_0 = tri[:,0]-tri[:,1]
    tri_1 = tri[:,1]-tri[:,2]
    tri_2 = tri[:,2]-tri[:,0]

    tri_0_norm = np.linalg.norm(tri_0, axis = -1)
    tri_1_norm = np.linalg.norm(tri_1, axis = -1)
    tri_2_norm = np.linalg.norm(tri_2, axis = -1)

    return ((tri_0_norm.mean()+ tri_1_norm.mean() + tri_2_norm.mean())/3)**2

def max_dist(tri):
    tri_0 = tri[:,0]-tri[:,1]
    tri_1 = tri[:,1]-tri[:,2]
    tri_2 = tri[:,2]-tri[:,0]

    tri_0_norm = np.linalg.norm(tri_0, axis = -1)
    tri_1_norm = np.linalg.norm(tri_1, axis = -1)
    tri_2_norm = np.linalg.norm(tri_2, axis = -1)

    return max((tri_0_norm.max(), tri_1_norm.max(), tri_2_norm.max()))**2


def calculate_dual_area(m,x):
    '''
    m is the center of the circle
    and x= [[a1,b1],....]
    '''
    a,b = x[:,0], x[:,1]
    area_1 = np.linalg.norm(np.cross(a,m), axis = -1)
    area_2 = np.linalg.norm(np.cross(b,m), axis = -1)
    return (area_1 + area_2)/4


def make_cotan_matrix(tri, tri_ind, num_vert):

    angle_2 = tri[:,[0,1]] - tri[:,np.newaxis, 2]
    angle_0 = tri[:,[1,2]] - tri[:,np.newaxis, 0]
    angle_1 = tri[:,[2,0]] - tri[:,np.newaxis, 1]


    cot_2 = calculate_cotangent(angle_2)/2
    cot_1 = calculate_cotangent(angle_1)/2
    cot_0 = calculate_cotangent(angle_0)/2

    # Make weight Matrix
    v0, v1, v2 = tri_ind[:,0], tri_ind[:,1], tri_ind[:,2]
    row = np.hstack([v0,v1,v0,v1,v1,v2,v1,v2,v2,v0,v2,v0])
    col = np.hstack([v1,v0,v0,v1,v2,v1,v1,v2,v0,v2,v2,v0])
    data = np.hstack([cot_2,cot_2,-cot_2,-cot_2
                      ,cot_0,cot_0,-cot_0,-cot_0
                      ,cot_1,cot_1,-cot_1,-cot_1])
    
    w_mat = sparse.coo_matrix((data,(row,col)), shape = (num_vert,num_vert), dtype = np.float64)

    return w_mat.tocsc()

def make_area_matrix_centric(tri, tri_ind, num_vert):
    
    angle_2 = tri[:,[0,1]] - tri[:,np.newaxis, 2]
    angle_0 = tri[:,[1,2]] - tri[:,np.newaxis, 0]
    angle_1 = tri[:,[2,0]] - tri[:,np.newaxis, 1]

    center = calculate_circumcenter(angle_2)+tri[:,2]
    area_0 = calculate_dual_area(center- tri[:,0],angle_0)
    area_1 = calculate_dual_area(center- tri[:,1],angle_1)
    area_2 = calculate_dual_area(center- tri[:,2],angle_2)
    
    v0, v1, v2 = tri_ind[:,0], tri_ind[:,1], tri_ind[:,2]
    row = np.hstack([v0,v1,v2])
    data = np.hstack([area_0,area_1,area_2])
    
    av_mat = sparse.coo_matrix((data,(row,row)), shape = (num_vert,num_vert), dtype = np.float64)
    return av_mat.tocsc()

def make_area_matrix_bary(tri, tri_ind, num_vert):
    angle_2 = tri[:,[0,1]] - tri[:,np.newaxis, 2]
    
    area = np.linalg.norm(np.cross(angle_2[:,0],angle_2[:,1]), axis = -1)/(2*3)

    v0, v1, v2 = tri_ind[:,0], tri_ind[:,1], tri_ind[:,2]
    row = np.hstack([v0,v1,v2])
    data = np.hstack([area,area,area])

    a_sparse = sparse.coo_matrix((data,(row,row)), shape = (num_vert,num_vert), dtype = np.float64)
    return a_sparse


def discrete_gradient(mesh, f):
    tri_ind = mesh.triangles
    tri  =mesh.vertices[tri_ind]

    e_2 = tri[:,1]-tri[:,0]
    e_0 = tri[:,2]-tri[:,1]
    e_1 = tri[:,0]-tri[:,2]

    normal = np.cross(e_2,-e_1)
    area = (np.linalg.norm(normal, axis = -1, keepdims = True))
    normal_unit = normal/area

    cross_2 = np.cross(normal_unit, e_2)
    cross_1 = np.cross(normal_unit, e_1)
    cross_0 = np.cross(normal_unit, e_0)

    grad_u_2 = cross_2*f[tri_ind[:,2], np.newaxis]
    grad_u_0 = cross_0*f[tri_ind[:,0], np.newaxis]
    grad_u_1 = cross_1*f[tri_ind[:,1], np.newaxis]
    
    grad_u = (grad_u_1+grad_u_2+grad_u_0)/(area)
    return grad_u

def discrete_laplacian(mesh, mode = 'Centric'):
    num_vert = len(mesh.vertices)
    tri_ind = mesh.triangles
    tri  =mesh.vertices[tri_ind]
    
    cotan_mat = make_cotan_matrix(tri,tri_ind, num_vert)

    if mode == 'Centric':
        area_mat = make_area_matrix_centric(tri, tri_ind, num_vert)
    elif mode == 'Bary':
        area_mat = make_area_matrix_bary(tri, tri_ind, num_vert)

    return cotan_mat, area_mat

def make_rotation_matrix(mesh):
    row = []
    col = []
    data = []
    
    num_vert = len(mesh.vertices)
    for i in range(num_vert):
        chart_i = mesh.chart(i)
        ind_i = chart_i['sort_ind'][:-1]
        a_i = chart_i['angles'][:-1]
        row.append(i)
        col.append(i)
        data.append(1)
        for n, j in enumerate(ind_i):
            chart_j = mesh.chart(j)
            a_j = chart_j['angles']
            m = np.where(np.array(chart_j['sort_ind']) == i)[0][0]
            
            a_ij = a_i[n]
            a_ji = a_j[m]
            
            row.append(i)
            col.append(j)
            data.append(np.exp(1j*(a_ij-a_ji+np.pi)))
        
    a_mat = sparse.coo_matrix((data,(row,col)), shape = (num_vert,num_vert), dtype = np.csingle)
    return a_mat

def discrete_connection_laplacian(mesh, mode = 'Centric'):
    rot_mat = make_rotation_matrix(mesh)
    cot_mat, area_mat = discrete_laplacian(mesh, mode)
    disc_conn_laplace = rot_mat.multiply(cot_mat)
    return disc_conn_laplace, area_mat



