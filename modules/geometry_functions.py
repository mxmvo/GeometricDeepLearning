import numpy as np
from scipy.sparse import linalg as splinalg

from modules.ddg import make_area_matrix_bary, make_area_matrix_centric, make_cotan_matrix, discrete_laplacian

'''
Given the simplicial complex, we will first calculate the eigenfunctions of the discrete laplace beltrami operator

Afterwards we will work on the heat kernel, wave kernel and optimal spectral descriptor. These are build using the eigenfunctions and eigenvalues.
'''
eps = np.finfo(np.float64).eps

b0 = lambda x: x**3/6
b1 = lambda x: (-3*x**3 +12*x**2 -12 *x +4)/6
b2 = lambda x: (3*x**3-24*x**2 +60*x -44)/6
b3 = lambda x: (-x**3+12*x**2-48*x+64)/6
def B_unit(x):
    y = x.copy()
    y[ x < 0] = 0
    y[ (0<= x)  & (x < 1) ] = b0(x[ (0<= x) & (x < 1) ])
    y[ (1<= x)  & (x < 2) ] = b1(x[ (1<= x) & (x < 2) ])
    y[ (2<= x)  & (x < 3) ] = b2(x[ (2<= x) & (x < 3) ])
    y[ (3<= x)  & (x < 4) ] = b3(x[ (3<= x) & (x < 4) ])
    y[ x >= 4] = 0
    return y


def B_spline_matrix(x, v_max, num_descr):
    B_descr = np.linspace(0,v_max,num_descr)
    B_delta = B_descr[1]-B_descr[0]
    x /= B_delta
    y = np.array([x - i +2 for i in range(num_descr)]).reshape(-1)
    y = B_unit(y).reshape(num_descr,-1) 
    return y

def laplace_beltrami_eigenfunctions(trimesh, mode = 'Centric' ,**kwargs):
    #tri = trimesh.vertices[trimesh.triangles]
    #tri_ind = trimesh.triangles
    #num_vert = len(trimesh.vertices)


    L, A = discrete_laplacian(trimesh, mode)
    
    val, vec = splinalg.eigsh(A, M= -L, which = 'LM', mode = 'buckling', sigma = -100, **kwargs)
    
    # smallest values are in the beginning and they become the biggest value
    # Why we need to do this because numberically the biggest (in magnitude) value can be negative
    sort = np.argsort(np.abs(val))[::-1]
    val = 1/val[sort]
    vec = vec[:,sort]

    return val, vec

def geometry_functions(mesh, v_max = 2100, num_descr = 150 , k = 300):
    val, vec = laplace_beltrami_eigenfunctions(mesh, k = k)
    B_mat = B_spline_matrix(val, v_max, num_descr)

    geom_func = np.matmul(B_mat, np.multiply(vec.T,vec.T))
    geom_func = geom_func/(np.linalg.norm(geom_func, axis =0, keepdims = True))
    return geom_func

'''


####
# Make the geometry vectors using a cardinal B-spline basis
####

b0 = lambda x: x**3/6
b1 = lambda x: (-3*x**3 +12*x**2 -12 *x +4)/6
b2 = lambda x: (3*x**3-24*x**2 +60*x -44)/6
b3 = lambda x: (-x**3+12*x**2-48*x+64)/6
def B_unit(x):
    y = x.copy()
    y[ x < 0] = 0
    y[ (0<= x)  & (x < 1) ] = b0(x[ (0<= x) & (x < 1) ])
    y[ (1<= x)  & (x < 2) ] = b1(x[ (1<= x) & (x < 2) ])
    y[ (2<= x)  & (x < 3) ] = b2(x[ (2<= x) & (x < 3) ])
    y[ (3<= x)  & (x < 4) ] = b3(x[ (3<= x) & (x < 4) ])
    y[ x >= 4] = 0
    return y
    
    
def B_spline_matrix(x, v_max, num_descr):
    B_descr = np.linspace(0,v_max,num_descr)
    B_delta = B_descr[1]-B_descr[0]
    x /= B_delta
    y = np.array([x - i for i in range(-2,num_descr-2)]).reshape(-1)
    y = B_unit(y).reshape(num_descr,-1)
    
    return y

def geometry_functions(eig_val, eig_vec, num_descr = None, v_max = None):
    if v_max == None:
        v_max = max(eig_val)

    if num_descr == None:
        num_descr = len(eig_val)//2
    b_mat = B_spline_matrix(eig_val, v_max, num_descr)

    geom =  np.matmul(b_mat, np.multiply(eig_vec, eig_vec).T)
    geom = geom/np.linalg.norm(geom, axis = 0, keepdims = True)
    return geom, b_mat


def make_weight_matrix_old(tri, tri_ind, num_vert):

    angle_2 = tri[:,[0,1]] - tri[:,np.newaxis, 2]
    angle_0 = tri[:,[1,2]] - tri[:,np.newaxis, 0]
    angle_1 = tri[:,[2,0]] - tri[:,np.newaxis, 1]


    cot_2 = calculate_cotangent(angle_2)/2
    cot_1 = calculate_cotangent(angle_1)/2
    cot_0 = calculate_cotangent(angle_0)/2

    # Make weight Matrix
    w_mat = np.zeros((num_vert, num_vert))

    w_mat[tri_ind[:,0],tri_ind[:,1]] += cot_2
    w_mat[tri_ind[:,1],tri_ind[:,0]] += cot_2

    w_mat[tri_ind[:,1],tri_ind[:,2]] += cot_0
    w_mat[tri_ind[:,2],tri_ind[:,1]] += cot_0

    w_mat[tri_ind[:,2],tri_ind[:,0]] += cot_1
    w_mat[tri_ind[:,0],tri_ind[:,2]] += cot_1

    diag = np.arange(num_vert)
    w_mat[diag,diag] = -w_mat.sum(-1)

    #print('Weight Matrix Density:', np.sum([w_mat != 0])/(num_vert**2))

    return sparse.csc_matrix(w_mat)

def make_area_matrix_bary_old(tri, tri_ind, num_vert):
    angle_2 = tri[:,[0,1]] - tri[:,np.newaxis, 2]
    
    a_mat = np.zeros((num_vert, num_vert))
    area = np.linalg.norm(np.cross(angle_2[:,0],angle_2[:,1]), axis = -1)/2

    a_mat[tri_ind[:,0], tri_ind[:,1]] += area
    a_mat[tri_ind[:,1], tri_ind[:,2]] += area
    a_mat[tri_ind[:,2], tri_ind[:,0]] += area
    a_mat = a_mat.sum(-1)/3
    a_sparse = sparse.csc_matrix(np.diag(a_mat))

    return a_sparse

def make_area_matrix_centric_old(tri, tri_ind, num_vert):
    
    angle_2 = tri[:,[0,1]] - tri[:,np.newaxis, 2]
    angle_0 = tri[:,[1,2]] - tri[:,np.newaxis, 0]
    angle_1 = tri[:,[2,0]] - tri[:,np.newaxis, 1]

    center = calculate_circumcenter(angle_2)+tri[:,2]
    area_0 = calculate_dual_area(center- tri[:,0],angle_0)
    area_1 = calculate_dual_area(center- tri[:,1],angle_1)
    area_2 = calculate_dual_area(center- tri[:,2],angle_2)
    
    av_mat = np.zeros((num_vert,num_vert))
    av_mat[tri_ind[:,0],tri_ind[:,1]] += area_0
    av_mat[tri_ind[:,1],tri_ind[:,2]] += area_1
    av_mat[tri_ind[:,2],tri_ind[:,0]] += area_2
    av_mat = av_mat.sum(-1)

    a_sparse = sparse.csc_matrix(np.diag(av_mat))

    return a_sparse

'''