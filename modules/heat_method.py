import scipy, sys
import scipy.sparse.linalg

import numpy as np

from modules.ddg import make_area_matrix_centric, make_area_matrix_bary, make_cotan_matrix, calculate_cotangent


eps = np.finfo(np.float64).eps

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

def discrete_divergence(e_1, e_2, c_1, c_2, X):
    term_1 = c_1*np.multiply(e_1,X).sum(-1)
    term_2 = c_2*np.multiply(e_2,X).sum(-1)
    return (term_1+term_2)/2


class heat_method():
    def __init__(self, vertices, triangles, L_c = None, A = None, t_mode = 'mean'):
        self.n = len(vertices)
        self.tri = vertices[triangles]
        self.tri_ind = triangles

        if L_c == None:
            L_c = make_cotan_matrix(self.tri, self.tri_ind, self.n)

        if A == None:
            A = make_area_matrix_centric(self.tri, self.tri_ind, self.n)

        self.A = A
        self.L_c = L_c

        if t_mode =='max':
            self.h2 = max_dist(self.tri)
        elif t_mode == 'mean':
            self.h2 = mean_dist(self.tri)

        self.M = {1:scipy.sparse.linalg.factorized(A-self.h2*L_c)}

        try:
            self.L = scipy.sparse.linalg.factorized(L_c)
        except RuntimeError:
            # The matrix is singular and cannot be factorized
            print('Matrix in singular, cannot continue')
            
        # The edges from the mesh, in a ordering
        # A triangle is given by v0 v1 v2, then v1-v0 = e2
        # in other words the edge across from the point.
        self.e_2 = self.tri[:,1]-self.tri[:,0]
        self.e_0 = self.tri[:,2]-self.tri[:,1]
        self.e_1 = self.tri[:,0]-self.tri[:,2]
        # Calculate the cross product for the gradient discretisation
        self.cross_0, self.cross_1, self.cross_2 = self.cross_grad()

        # Precalculate the cotangents
        self.cot_0 ,self.cot_1, self.cot_2 = self.calc_cotangents()

    def calc_cotangents(self):
        c_0 = self.tri[:,[1,2]]-self.tri[:,np.newaxis, 0]
        c_1 = self.tri[:,[0,2]]-self.tri[:,np.newaxis, 1]
        c_2 = self.tri[:,[0,1]]-self.tri[:,np.newaxis, 2]

        cot_0 = calculate_cotangent(c_0)
        cot_1 = calculate_cotangent(c_1)
        cot_2 = calculate_cotangent(c_2)

        return cot_0, cot_1, cot_2


    def cross_grad(self):
        n_1 = self.tri[:,1]-self.tri[:,0]
        n_2 = self.tri[:,2]-self.tri[:,0]
        normal = np.cross(n_1,n_2)
        area = (np.linalg.norm(normal, axis = -1, keepdims = True))
        normal_unit = normal/area

        cross_2 = np.cross(normal_unit, self.e_2)/(area)
        cross_1 = np.cross(normal_unit, self.e_1)/(area)
        cross_0 = np.cross(normal_unit, self.e_0)/(area)

        return cross_0, cross_1, cross_2

    def run(self, ind, m = 1):
        # Step 1 calculate a solution to the heat flow
        u_0 = np.zeros(self.n)
        u_0[ind] = 1
        
        # Solve
        if m not in self.M.keys():
            self.M[m] = scipy.sparse.linalg.factorized(self.A-m*self.h2*self.L_c)

        mat = self.M[m]
        u_t = mat(u_0)
        
        
        # Step two in the heat method
        # Calculate gradient
        grad_u_2 = self.cross_2*u_t[self.tri_ind[:,2], np.newaxis]
        grad_u_0 = self.cross_0*u_t[self.tri_ind[:,0], np.newaxis]
        grad_u_1 = self.cross_1*u_t[self.tri_ind[:,1], np.newaxis]
        
        grad_u = (grad_u_1+grad_u_2+grad_u_0)
        # Normalise

        X = -grad_u/(np.linalg.norm(grad_u, axis = -1, keepdims = True))
        
        # Step three in the heat method
        
        div_0 = discrete_divergence(self.e_2, -self.e_1, self.cot_2, self.cot_1, X)
        div_1 = discrete_divergence(self.e_0, -self.e_2, self.cot_0, self.cot_2, X)
        div_2 = discrete_divergence(self.e_1, -self.e_0, self.cot_1, self.cot_0, X)
        
        div_X = np.zeros(self.n)

        for i, tri in enumerate(self.tri_ind):
            i_0, i_1, i_2 = tri
            div_X[i_0] += div_0[i]
            div_X[i_1] += div_1[i]
            div_X[i_2] += div_2[i]

        
        phi = self.L(div_X)
        phi = phi-phi.min()
        return phi, X
