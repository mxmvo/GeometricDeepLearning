import numpy as np
import time

'''
The idea is that we have a function that you can give the initial point with the mesh. It will then return the vertices with values

For now we assume that all the angles are acute, i.e. no triangle has angels bigger then 90 degrees, and that the 'force' is constantly one
'''



class FMM():
    def __init__(self,trimesh):
        self.mesh = trimesh

        tri = self.mesh.vertices[self.mesh.triangles].astype(np.float)
        e0 = tri[:,2] - tri[:,1]
        e1 = tri[:,0] - tri[:,2]
        e2 = tri[:,1] - tri[:,0]
        normal = np.cross(e2,-e1)
        self.area = np.linalg.norm(normal, axis = -1, keepdims = True)
        self.normal = normal/self.area
        self.mat = np.stack([e0,e1,e2], axis = 1)/self.area[:,:,np.newaxis]
        self.inner_mat = np.einsum('ijk,ilk->ijl', self.mat, self.mat)

        self.adj_ind = {}


    def add_points_to_near(self, new_alive_points):
        added_points = set()
        for point in new_alive_points:
            tri = self.mesh.triangles_containing_node(point)
            for t in tri:
                last_vertex = list( (set(t) - self.alive) )

                if len(last_vertex) == 1:
                    self.near_values[last_vertex[0]] = np.inf
                    self.near.add(last_vertex[0])
                    added_points.add(last_vertex[0])
        return added_points

    def update_values_in_near_discrete_grad(self, added_points):

        for point in added_points:
            #print(point)
            if point not in self.adj_ind:
                self.adj_ind[point]  = np.where(self.mesh.triangles == point)[0]

            adj_tri_ind = self.adj_ind[point]


            for tri_ind in adj_tri_ind:
                
                tri = self.mesh.triangles[tri_ind]
                
                i = np.where(tri == point)[0][0]
                j = (i+1) % 3
                k = (i+2) % 3
                l_edge = [tri[j],tri[k]]
                edge = set(l_edge)


                if edge.issubset(self.alive):
                    d_j, d_k = self.alive_values[l_edge[0]], self.alive_values[l_edge[1]]
                    mat = self.inner_mat[tri_ind]
                    triangle_area = self.area[tri_ind][0]

                    sq_term = mat[i,i]
                    lin_term = 2*(d_j*mat[i,j]+d_k*mat[i,k])
                    con_term = (d_j**2)*mat[j,j] + (d_k**2)*mat[k,k] + 2*d_j*d_k*mat[j,k]-1

                    det = lin_term**2-4*sq_term*con_term

                    if det >= 0:
                        det_sqr = np.sqrt(det)
                        root_1, root_2 = (-lin_term + np.array([- det_sqr, det_sqr]))/(2*sq_term)
                        if root_1 > root_2:
                            print('wtf')
                            root_1, root_2 = root_2, root_1

                        pif = False
                        if root_1 >= max(d_j, d_k):
                            d_i = root_1
                            pif = True
                        elif root_2 >= max(d_j, d_k):
                            d_i = root_2
                            pif = True
                        else:
                            d_i = min(d_k + triangle_area*np.sqrt(mat[j,j]), d_j+ triangle_area*np.sqrt(mat[k,k]))


                        if pif:
                            # some check point is updated from within the triangle
                            e_mat = self.mat[tri_ind]
                            

                            sum_e = d_i*e_mat[i]+d_j*e_mat[j]+d_k*e_mat[k]

                            grad = np.cross(self.normal[tri_ind], sum_e)

                            a,b = self.mesh.decompose_vector_3D(-grad, -e_mat[j], e_mat[k])
                            #print(a,b)
                            if ~ ((a > 0) and (b> 0)):
                                d_i = min(d_k + triangle_area*np.sqrt(mat[j,j]), d_j+ triangle_area*np.sqrt(mat[k,k]))
                    else:
                        d_i = min(d_k + triangle_area*np.sqrt(mat[j,j]), d_j+ triangle_area*np.sqrt(mat[k,k]))


                    #print(self.near_values[point], d_i)
                    if d_i < self.near_values[point]:
                        self.near_values[point] = d_i
 

    def update_values_in_near_original(self, added_points):

        for point in added_points:
            neighbors = self.mesh.chart(point)['sort_ind']#sorted_neigh_vertex(point)
            val = []
            indices = []

            for i in range(1,len(neighbors)):

                edge_a, edge_b = neighbors[i-1], neighbors[i]
                edge = {edge_a, edge_b}

                if edge.issubset(self.alive):
                    T_a, T_b = self.alive_values[edge_a], self.alive_values[edge_b]
                    if T_a >= T_b:
                        T_a, T_b = T_b, T_a
                        edge_a, edge_b = edge_b, edge_a
                    indices.append([edge_a,edge_b,point])
                    val.append([T_a,T_b])

            val = np.array(val)
            if len(indices) == 0:
                continue
            indices = np.array(indices)
            T_c = self.calculate_T_c(indices,val)
            min_ind = np.argmin(T_c)

            if T_c[min_ind] < self.near_values[point]:
                self.near_values[point] = T_c[min_ind]



    def calculate_T_c(self,indices,val):
        points = self.mesh.vertices[indices]

        T_c = self.calculate_t(points[:,0],points[:,1],points[:,2],val[:,0],val[:,1])
        return T_c

    def calculate_t(self,A,B,C,T_a, T_b, F = 1):
        assert (T_b >= T_a).all(), 'T_b should be bigger'
        ac = np.linalg.norm(A-C, axis = 1, keepdims = True)
        bc = np.linalg.norm(B-C, axis = 1, keepdims = True)
        cos_theta = np.multiply((A-C)/ac,(B-C)/bc).sum(axis = 1)
        theta = np.arccos(cos_theta)
        u = (T_b - T_a)
        ac = ac.reshape(-1)
        bc = bc.reshape(-1)
        cos_theta = cos_theta.reshape(-1)



        sq_term = ac**2+bc**2 - 2*cos_theta*ac*bc
        lin_term = 2*u*ac*(cos_theta*bc-ac)
        con_term = -(F*np.sin(theta)*ac*bc)**2 + (u*ac)**2


        det = lin_term**2-4*sq_term*con_term
        det_neg= det < 0
        det_sqr = np.sqrt(det[~ det_neg])

        roots = np.zeros((2,len(det)))
        abc_roots = (-lin_term[~ det_neg] + np.array([- det_sqr, det_sqr]))/(2*sq_term[~ det_neg])
        #print(roots.shape, roots[:, ~ det_neg], abc_roots, det, det_sqr)
        roots[:, ~ det_neg] = abc_roots

        roots[:,det_neg] = -1

        roots = np.sort(roots, axis = 0)

        criterions = ac*(1-u/roots)

        b_mat = ((cos_theta*bc) < criterions) & ( criterions< (bc/cos_theta)) & (u < roots)

        t_c = np.zeros(roots.shape[1])
        for i, b in enumerate(b_mat.T):
            if b.sum() > 0:
                t =T_a[i] + roots[b][0][i]
            else:
                t = min(T_a[i] + ac[i]*F, T_b[i] + bc[i]*F)
            t_c[i] = t

        return t_c

    def _run_init(self, start_ind):
        '''
        Make the necessary sets, and populate the initial values.
        '''
        self.start_ind = start_ind

        self.alive_values = {}
        self.alive = set()
        self.alive_mode = {}

        self.near = set()
        self.near_values = {}
        self.near_mode = {}
        for ind in start_ind:
            self.alive_values[ind] = 0
            self.alive.add(ind)

            neighbors = list(self.mesh.get_neighbors_unsorted(ind)- {ind})
            num_neighbors = len(neighbors)

            new_alive_points = []
            for n in range(1,num_neighbors):
                point = neighbors[n]
                new_alive_points.append(point)

                self.alive.add(point)
                self.alive_values[neighbors[n]] = np.linalg.norm(self.mesh.vertices[ind]-self.mesh.vertices[point])

            added_points = self.add_points_to_near(new_alive_points)
            self.update_values_in_near_discrete_grad(added_points)

    def run(self, start_ind = [], max_distance = np.inf):
        self._run_init(start_ind)
        front_max = 0
        
        while (front_max < max_distance) and (len(self.near)>0):

            min_ind = min(self.near_values, key = self.near_values.get )


            # Add it to alive and remove it from near
            self.near.remove(min_ind)
            self.alive.add(min_ind)
            self.alive_values[min_ind] = self.near_values.pop(min_ind)

            # Update near
            added_near_points = self.add_points_to_near([min_ind])


            # Update T values
            self.update_values_in_near_discrete_grad(added_near_points)

            front_max = self.alive_values[min_ind]

        a_func = self.alive_values
        n_func = self.near_values
        #max_val = max(self.alive_values.values())
        col = np.array([np.inf]*len(self.mesh.vertices))
        col[list(a_func.keys())] = list(a_func.values())
        col[list(n_func.keys())] = list(n_func.values())
        return col

