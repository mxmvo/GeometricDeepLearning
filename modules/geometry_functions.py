import numpy as np
from scipy import sparse

'''
Given the simplicial complex, we will first calculate the eigenfunctions of the discrete laplace beltrami operator

Afterwards we will work on the heat kernel, wave kernel and optimal spectral descriptor. These are build using the eigenfunctions and eigenvalues.
'''
eps = np.finfo(np.float64).eps

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


def calculate_dual_area(m,x):
    '''
    m is the center of the circle
    and x= [[a1,b1],....]
    '''
    a,b = x[:,0], x[:,1]
    area_1 = np.linalg.norm(np.cross(a,m), axis = -1)
    area_2 = np.linalg.norm(np.cross(b,m), axis = -1)
    return (area_1 + area_2)/4


def make_weight_matrix(tri, tri_ind, num_vert):

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
    
    a_mat = np.zeros(num_vert)
    area = np.linalg.norm(np.cross(angle_2[:,0],angle_2[:,1]), axis = -1)/(2*3)

    v0, v1, v2 = tri_ind[:,0], tri_ind[:,1], tri_ind[:,2]
    row = np.hstack([v0,v1,v2])
    data = np.hstack([area,area,area])

    a_sparse = sparse.coo_matrix((data,(row,row)), shape = (num_vert,num_vert), dtype = np.float64)
    return a_sparse


def laplace_beltrami_eigenfunctions(trimesh, mode = 'Centric' ,**kwargs):
    tri = trimesh.vertices[trimesh.triangles]
    tri_ind = trimesh.triangles
    num_vert = len(trimesh.vertices)

    w_sparse= make_weight_matrix(tri, tri_ind, num_vert)

    # Make area matrix
    if mode == 'Centric':
        a_sparse = make_area_matrix_centric(tri, tri_ind, num_vert)
    elif mode == 'Bary':
        a_sparse = make_area_matrix_bary(tri, tri_ind, num_vert)

    print('Calculation Eigenfunctions/values ...')
    
    val, vec = sparse.linalg.eigsh(a_sparse, M= w_sparse, which = 'LM', mode = 'buckling', **kwargs)
    
    # smallest values are in the beginning and they become the biggest value
    # Why we need to do this because numberically the biggest (in magnitude) value can be negative
    sort = np.argsort(np.abs(val))[::-1]
    val = 1/val[sort]
    vec = vec[:,sort]

    return val, vec

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

'''
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