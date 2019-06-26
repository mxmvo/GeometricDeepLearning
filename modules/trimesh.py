import numpy as np
import matplotlib.cm as cm
import time

from .config import trimesh_config

eps = np.finfo(np.float64).eps
class trimesh():
    '''
    vertices : np.array [[x1,y1,z1],...,[xn,yn,zn]]
        x,y,z are coordinates in R^3
    triangles: np.array [[a1,b1,c1],...,[am,bm,cm]]
        a,b,c are indices from the vertices
    '''
    
    def __init__(self,vertices, triangles):
        self.vertices = np.array(vertices)
        self.triangles = np.array(triangles)

        # Dictionairy that will save the computed atlases of nbhs
        self._atlas = {}


    '''
    Query functions
    '''

    def get_neighbors(self,ind, start = None):
        '''
        inp: ind an array of indices ind_1
        out: [n_11,...,n_1a, n_11]

        todo: See if I can vectorize this in a way
        '''
        n = [ind]
        
        triangles = self.triangles

        adj_tri = triangles[(triangles == ind).sum(-1) > 0]

        if start == None:
            i = [0]
        else:
            i = np.where(adj_tri == start)[0]
        
        available_triangles = list(range(len(adj_tri)))

        while len(available_triangles) > 0:
            i = available_triangles.pop(i[0])
            tri = adj_tri[i]

            tri = [v for v in tri if ((v != ind) and (v != n[-1]))]
            n.append(tri[0])

            mat = ( adj_tri[available_triangles] == n[-1] ).sum(-1)

            i = np.where(mat == 1)[0]



        n = n[1:]
        n.append(n[0])
        return n

    def get_neighbors_unsorted(self,ind):
        adj_tri = self.triangles_containing_node(ind)
        return set(adj_tri.reshape(-1))

    def common_neighbors(self,arr):
        return set.intersection(*[self.get_neighbors_unsorted(v) for v in arr])


    def triangles_containing_node(self,ind):
        tri  = self.triangles
        adj_tri = tri[np.where(tri == ind)[0]]
        return adj_tri

    def triangles_containing_edge(self,e):
        tri  = self.triangles
        adj_tri = tri[np.asarray(np.isin(tri, e).sum(1) == 2).nonzero()[0]]
        
        #adj_tri = tri[(tri[:,:,np.newaxis] == np.array([[e]])).sum(-1).sum(-1) == 2]
        return adj_tri
    
    def triangles_containing_edges(self,e1,e2):
        tri  = self.triangles
        adj_tri = tri[(tri[:,:,np.newaxis] == np.array([[list(e1) + list(e2)]])).sum(-1).sum(-1) == 4]
        return adj_tri    

    def sorted_neigh_edge_point(self,path,triangles):
        assert len(triangles) == 2, 'Assumption edge point violated'
        previous_edge = path['edges'][-2]
        current_edge = path['edges'][-1]
        
        start_face = path['faces'][-1]

        i_start = np.where((triangles[:,:,np.newaxis] == start_face.reshape(1,1,-1)).sum(-1).sum(-1) == 3)[0]
        i_other = (i_start+1)%2

        other_face = triangles[i_other][0]
    
        
        if previous_edge[0] == previous_edge[1]:
            p1 = {current_edge[0]}
        else:
            p1 = set(previous_edge) & set(current_edge)
        
        p3 = set(current_edge) - p1

        p2 = set(other_face) - set(current_edge)
        p4 = set(previous_edge) - p1

        return  [list(p1)[0],list(p2)[0],list(p3)[0],list(p4)[0]]

    def sorted_neigh_vertex(self, ind):
        adj_tri = self.triangles_containing_node(ind)
        boundary = [set(tri) - {ind} for tri in adj_tri]
        n = list(boundary[0])
        for _ in range(len(boundary)):
            for e in boundary:
                if e.issubset(set(n)):
                    continue

                next_ind = e.difference(set(n))
                
                if len(next_ind) == 1:
                    common_ind = list(e.intersection(set(n)))[-1]
                    if common_ind == n[-1]:
                        n.append(list(next_ind)[0])
                    else:
                        n.insert(0,list(next_ind)[0])
                    break
        if {n[0],n[-1]} in boundary:
            n.append(n[0])
        return n

    def face_ind(self,face):
        f = list(face)
        i0 = np.where(self.triangles == f[0])[0]
        i1 = np.where(self.triangles == f[1])[0]
        i2 = np.where(self.triangles == f[2])[0]
        s = set.intersection(set(i0),set(i1),set(i2))        
        assert len(s)== 1, 'It should be 1'
        return s.pop()
    
    '''
    Functions for the calculating the neighborhood
    '''
    def chart(self, ind):
        '''
        Return the neighborhood around an index
        '''
        self.make_atlas(indices = [ind])
        return self._atlas[ind]

    def make_atlas(self, indices = [], scale = 1):
        '''
        make chart around a list of indices.
        '''
        if len(indices) == 0:
            indices = list(range(len(self.vertices)))
        
        for i in indices:
            if i in self._atlas:
                continue
            hood, ind = self.make_chart(i, scale = scale)
            self._atlas[i] = {'nbh':hood,'sort_ind':ind}

    def make_chart(self, center_vertex, scale = 1):
        '''
        Make 2-dimensional neighborhood around the vertex

        center_vertex: the point around which we want to make the neighborhood
        '''
        center_pnt = self.vertices[center_vertex]
        
        # Get the neighbors in sorted order
        nbh_ind = self.sorted_neigh_vertex(center_vertex)
        nbh_pnt = self.vertices[nbh_ind]
        
        # Calculate the angle between adjacent neighbors
        angles = self.calculate_angles(center_pnt, nbh_pnt)

        # If on the boundary map to half plane
        # Else to the plane
        if nbh_ind[0] != nbh_ind[-1]:
            angles = np.hstack([[0], np.pi*angles[:-1]/angles[-2]])
        else:
            angles = np.hstack([[0], angles[:-1]])
        
        pnt_centered = nbh_pnt-center_pnt
        norm_pnt = np.linalg.norm(pnt_centered, axis = 1, keepdims = True)

        pnts_2D = np.c_[np.cos(angles),np.sin(angles)]
        pnts_2D = pnts_2D*(norm_pnt*scale)

        nbh_hood = {}
        for i, n in enumerate(nbh_ind):
            nbh_hood[n] = pnts_2D[i]
        
        return nbh_hood, nbh_ind

    def calculate_angles(self,center,neigh_points):
        centered_neigh_mat = neigh_points-center
        norm_neigh_mat = centered_neigh_mat/(np.linalg.norm(centered_neigh_mat, axis = 1, keepdims = True))
        
        inner_neigh_mat = np.matmul(norm_neigh_mat,norm_neigh_mat.T)
            
        size = len(inner_neigh_mat)
        
        angles = inner_neigh_mat[[np.arange(size)],[np.arange(-size+1,1)]]

        #start_first = inner_neigh_mat[0,1]
        #end_start = inner_neigh_mat[-1,0]
        #first_end = inner_neigh_mat[1,-1]
        
        #test_angles = np.arccos(np.clip([start_first,end_start, first_end],-1,1))
        #print(test_angles, np.allclose(test_angles[-1], test_angles[0]+test_angles[1]))

        angles = np.arccos(np.clip(angles,-1,1))
        

        #print(angles.sum()/(2*np.pi))
        angles = 2*np.pi*np.cumsum(angles)/angles.sum()
        
        return angles

    '''
    Computation functions
    '''

    def chart_to_mesh(self, pnts_2D, nbh_hood, nbh_ind,center_vertex):
        center = self.vertices[center_vertex]
        pnts_3D = []
        for pnt in pnts_2D:
            if nbh_ind[0] != nbh_ind[-1]:
                pnt[1] = max(np.finfo(float).eps,pnt[1])
            edge = list(self.which_edge(pnt,nbh_hood,nbh_ind))
            v1, v2 = nbh_hood[edge[0]], nbh_hood[edge[1]]
            a,b = self.decompose_vector(pnt,v1, v2)
            

            V1, V2 = self.vertices[edge[0]]-center, self.vertices[edge[1]]-center
            pnts_3D.append(center+a*V1+b*V2)
        
        return np.array(pnts_3D)




    def mesh_to_chart(self, point, face, nbh_hood):
        '''
        Transforms a 3D point to the 2D equivalent in the neighborhood

        Arguments:
        point: the 3d point to transform
        face: the face in which the point lies
        nbh_hood: the neighborhood to map the point to. 
        '''
        indices = set(nbh_hood.keys())
        face = set(face)
        if len(list(indices.intersection(face))) < 2:
            print(indices, face, point)
        [n1, n2] = list(indices.intersection(face))
        
        center_ind = list(face.difference(indices))[0]

        center_point = self.vertices[center_ind]
        pnt_1, pnt_2 = self.vertices[n1]-center_point, self.vertices[n2]-center_point
        

        pnt_0 = point - center_point
        a, b = self.decompose_vector_3D(pnt_0, pnt_1 , pnt_2 )


        return a*nbh_hood[n1]+ b*nbh_hood[n2]
        

    def grad_hess_approx(self,value_function, nbh_function, center_vertex):
        '''
        f(x) = f(x0) + Df(x-x0) + (x-x0)^T Df^2/2 (x-x0)+ ....
        f(x) - f(x0) = [Df:Df^2]*[x,y, .5x^2, xy, .5y^2].T 
        '''
        indices = [i for i in nbh_function if i in value_function]
        values = np.array([value_function[i] for i in indices]) - value_function[center_vertex]
        points = np.array([nbh_function[i] for i in indices])

        if len(points) < 4:
            mat = points
        else:
            order_2nd = np.zeros((len(points),3))
            order_2nd[:,0] = .5*points[:,0]*points[:,0] 
            order_2nd[:,1] = points[:,0]*points[:,1] 
            order_2nd[:,2] = .5*points[:,1]*points[:,1] 

            mat = np.hstack((points, order_2nd))

        inverse = np.linalg.pinv(mat)
        func_approx = np.dot(inverse, values)
        return func_approx

    def grad_vertex_nbh(self,indices, X):
        X_vert = {}
        for i in indices:
            f_ind = np.where(self.triangles == i)[0]

            # map to 2d
            nbh = self.chart(i)['nbh']
            points_2d = []
            area = []
            for f in f_ind:
                face = self.triangles[f]
                v1, v2 = list(set(face)-{i})
                p1,p2 = self.vertices[[v1,v2]]- self.vertices[i]
                grad = X[f]
                a,b = self.decompose_vector_3D(grad,p1,p2)
                points_2d.append(a*nbh[v1]+b*nbh[v2])

                area.append(np.linalg.norm(np.cross(p1,p2)))

            points_2d = np.array(points_2d)
            area = np.array(area)

            flat_grad = np.multiply(points_2d,area[:,np.newaxis]).sum(0)
            flat_grad = flat_grad/(np.linalg.norm(flat_grad) + eps)
            X_vert[i] = flat_grad*0.00001
        return X_vert



    def which_edge(self, point, nbh_hood, nbh_ind):
        
        #if all(point == [0,0]):
        #   return nbh_ind[:2]
        for i in range(len(nbh_ind)-1):
            n1, n2  = nbh_ind[i], nbh_ind[(i+1)% len(nbh_ind)]
            
            vec1 = nbh_hood[n1]
            vec2 = nbh_hood[n2]
            
            if self.inside_triangle_check(point, vec1, vec2):
                return set([n1,n2])
        
    def inside_triangle_check(self, point, vec1, vec2):
        '''
        Given a triangle defined by v1 and v2 and (0,0). 
        Then the p is inside when:
        a,b, > 0 and a+b < 1
        with:
            a = (p x v2) / (v1 x v2)
            b = -(p x v1) / (v1 x v2)

            then p  = av1 + bv2
            u x v = u1v2-u2v1
        '''
        a,b = self.decompose_vector(point, vec1, vec2)
        
        if (a >= 0) and (b>= 0) and (a+b <= 1):
            
            return True
        else:
            return False

    def decompose_vector(self, point, vec1, vec2): 
        '''
        Given a triangle defined by v1 and v2 and (0,0). 
        Then p  = av1 + bv2
        with:
            a = (p x v2) / (v1 x v2)
            b = -(p x v1) / (v1 x v2)

            u x v = u1v2-u2v1
        '''
        
        den = vec1[0]*vec2[1] - vec1[1]*vec2[0]
        num_a = point[0]*vec2[1] - point[1]*vec2[0]
        num_b = point[0]*vec1[1]- point[1]*vec1[0]
        #print(den)
        a = num_a/den
        b = -num_b/den

        # Below does the same but it has a little more overhead and is therefor less efficient.
        #a,b = np.matmul(np.linalg.inv(np.stack([vec1, vec2], axis = 1)),point)
        return a, b

    def decompose_vector_3D(self, pnt_0, pnt_1, pnt_2): 
        '''
        Given a triangle defined by v1 and v2 and (0,0). 
        Then p  = av1 + bv2
        with:
            a = (p x v2) / (v1 x v2)
            b = -(p x v1) / (v1 x v2)

            u x v = u1v2-u2v1
        '''
        
        pnts = np.vstack([ [pnt_0], [pnt_1], [pnt_2]])
    
        pnts_norm = np.linalg.norm(pnts, axis =1, keepdims = True)
        pnts_unit = pnts/(pnts_norm +np.finfo(float).eps)
        vec1 = np.array([1,0])*pnts_norm[1,0]
        
        x = pnts_unit[1]
        y = pnts_unit[2]
        y = y-x*np.dot(x,y)
        y = y/np.linalg.norm(y)

        vec0 = np.array([np.dot(x,pnts[0]),np.dot(y,pnts[0])])
        vec2 = np.array([np.dot(x,pnts[2]),np.dot(y,pnts[2])])

        return self.decompose_vector(vec0,vec1,vec2)

    def gradient_field(self, indices, value_dict, atlas,  mode = 'descent', dim = '2'):
        assert (mode == 'descent') or (mode == 'ascent'), 'Mode should be ascent or descent'
        vector_field = []

        for ind in indices:
            #nbhs = self.get_neighbors(ind)

            center = self.vertices[ind]

            nbh_hood, nbh_ind= atlas[ind]['nbh'], atlas[ind]['sort_ind']
            grad_hess_approx = self.grad_hess_approx(value_dict, nbh_hood, ind)

            grad = grad_hess_approx[:2]
            if mode =='descent':
                grad *= -1
            

            grad = grad/np.linalg.norm(grad)*0.0001
            if dim == '3':            
                next_point = self.chart_to_mesh([grad],nbh_hood,nbh_ind,ind)[0]
                vector_field.append((next_point-center)/np.linalg.norm(next_point-center))
            
            elif dim == '2':
                vector_field.append(grad)
        return np.array(vector_field)

    

    def grad_hess_field(self, indices, value_function, atlas):
        field = {}

        for i in indices:
            field[i] = self.grad_hess_approx(value_function, atlas[i]['nbh'],i)

        return field

    def linear_field(self, indices, vector_field, atlas):
        field  = {}
        for i in indices:
            field[i] = self.linear_approx(vector_field, atlas[i]['nbh'], i)

        return field
    
    def linear_approx(self, vector_field, nbh, i):
        # find the faces
        # transform the center to 2 d
        # Calculate the gradient approx

        tri_ind = np.where(self.triangles == i)[0]
        points = []
        values = []
        for t  in tri_ind:
            face  = self.triangles[t]
            t_center = self.vertices[face].mean(0)
            p_2d = self.mesh_to_chart(t_center, face, nbh)
            points.append(p_2d)
            values.append(vector_field[t])

        Df = np.matmul(np.linalg.pinv(points), values)

        return Df

    def vtki_plot(self,ind, color, cmap = 'jet', paths = [], labels ={}, arrows_tri = [], arrows_vert = [], text = ''):
        polydata = PolyData(self.vertices, np.c_[[[3]]*len(self.triangles),self.triangles])
        plotter = vtki.BackgroundPlotter()

        plotter.add_mesh(polydata, scalars = color, cmap = cmap, show_edges = True)
        plotter.add_text(text)
        if len(ind)> 0:
            plotter.add_points(self.vertices[ind], point_size = 10, color = 'red')

        if len(labels)>0:
            plotter.add_point_labels(self.vertices[labels['pnts']], labels['labels'], font_size =20)

        if len(paths) > 0 :
            for pa in paths:
                p = np.array(pa['points'])
                if len(p) > 0:
                    plotter.add_lines(p, width = 1, color = 'black')
        
        if len(arrows_tri) > 1:
            cent = self.vertices[self.triangles].mean(1)
            plotter.add_arrows(cent, arrows_tri, mag  = .01, color = 'g')

        if len(arrows_vert) > 1:
            plotter.add_arrows(self.vertices, arrows_vert, mag  = 1, color = 'g')
        plotter.view_xy()


'''

 def barycentric_coord(self, pnt, vec1, vec2, vec3):
        
        If pnt lies in the triangle of which the point is vec1, vec2 and vec3
        then 
        pnt = a*vec1 + b*vec2 + c*vec3
        s.t. 0<a,b,c<1 and a+b+c = 1
        

        mat = np.vstack([vec1,vec2,vec3]).T
        para = np.dot(self.inversion(mat), pnt)
        return list(para)


    def inversion(self, m):    
        m1, m2, m3, m4, m5, m6, m7, m8, m9 = m.ravel()
        inv = np.array([[m5*m9-m6*m8, m3*m8-m2*m9, m2*m6-m3*m5],
                    [m6*m7-m4*m9, m1*m9-m3*m7, m3*m4-m1*m6],
                    [m4*m8-m5*m7, m2*m7-m1*m8, m1*m5-m2*m4]])
        return inv / np.dot(inv[0], m[:, 0])

def radial_paths(self,ind, n_ind = None, bins = 5):
        if n_ind == None:
            n_ind = np.random.choice(self.get_neighbors(ind))

        adj_tri = self.triangles_containing_edge([ind,ind])
        #current_face = self.triangles_containing_edge([ind,n_ind])[0]
        #path_info = [[[n_ind,n_ind], None],[[ind,ind], current_face]]
        #path = self.vertices[[n_ind,ind]]

        init_path = self.initialise_path(ind = n_ind, n_ind = ind)
        neighbors = self.sorted_neigh_vertex_path(init_path,adj_tri)

        translation = {i:neighbors[i] for i in range(len(neighbors))}
        
        neigh_points = np.r_[[init_path['points'][-2]],self.vertices[neighbors]]
        center = init_path['points'][-1]
        angles = self.calculate_angles(center, neigh_points)
        num_angles = len(angles)
        
        
        paths = [self.initialise_path(ind = ind, n_ind = n_ind)]
        for i in range(1,bins):
            a = i*2*np.pi/bins
            if (angles == a).sum() == 1:
                s_ind = np.where(angles == a)
                edge = [translation[s_ind],translation[s_ind]]
                next_point = self.vertices[edge[0]]
            else:
                s_ind = (np.where(angles > a)[0][0]) -1
                s_ind_next = (s_ind +1) % (num_angles)

                edge = [translation[s_ind], translation[s_ind_next]]

                ratio  = (a-angles[s_ind])/(angles[s_ind_next]-angles[s_ind])

                next_point = self.calculate_next_point(init_path, edge,ratio)
            paths.append(self.initialise_path(ind,point = next_point,edge = edge))

        return paths


 def walk_straight(self, path, max_length , stop_edges):
        # Do a certain number of steps
        path_length = 0
        while path_length < max_length:

            adj_tri = self.triangles_containing_edge(path['edges'][-1])
            
            # boundary check
            if len(adj_tri) == 1:
                print('Breaks')
                break

            # If the path ended at a node or on a edge
            if len(adj_tri) == 2:
                neighbors = self.sorted_neigh_edge_point(path,adj_tri)
            else:
                neighbors = self.sorted_neigh_vertex_path(path,adj_tri)

            
            # A translation from ind to matrix_ind

            # Make innerproduct matrix
            neigh_mat = np.r_[[path['points'][-2]],self.vertices[neighbors]]
            translation = {i:neighbors[i] for i in range(len(neighbors))}
            
            angles = self.calculate_angles(path['points'][-1], neigh_mat)
            # if end precisely on a point then take
            # that point to be the next (this will prop never happen)
            # otherwise we need to calculate the point
            if (angles == np.pi).sum() ==1:
                s_ind = np.where(angles == np.pi)[0][0]

                next_edge = [translation[s_ind],translation[s_ind]]
                next_point = self.vertices[next_edge[0]]
            else:
                s_ind = np.where(angles > np.pi)[0][0]
                next_edge = [translation[s_ind-1],translation[s_ind]]
            
            
                ratio = (np.pi-angles[s_ind-1])/(angles[s_ind]-angles[s_ind-1])
                next_point = self.calculate_next_point(path,next_edge,ratio)
            
            next_face = self.triangles_containing_edges(next_edge,path['edges'][-1])[0]
            
            path['points'] = np.append(path['points'], [next_point], axis = 0)
            
        
            path['edges'] = np.append(path['edges'],[next_edge], axis = 0)
            path['faces'] = np.append(path['faces'],[next_face], axis  = 0)

            path_length = np.linalg.norm((path['points'][1:]-path['points'][:-1]), axis = 1).sum()
            
            if set(next_edge) in stop_edges:
                break
        return path


   def initialise_path(self, ind, point = [], gradient = [], edge = [], n_ind = None):
        
        #Make path and path_info
        
        
        if (len(point) == 0 )& (len(gradient) == 0):
            if n_ind == None:
                n_ind = np.random.choice(self.get_neighbors(ind))
            path = self.vertices[[ind,n_ind]]
            edge = [n_ind,n_ind]
        elif len(gradient) > 0:
            assert len(edge) == 2, 'If given a point also provide an edge'
            path = np.r_[[self.vertices[ind]],[point]]
            grad = np.array([gradient])
            
        elif len(point) > 0:
            assert len(edge) == 2, 'If given a point also provide an edge'
            path = np.r_[[self.vertices[ind]],[point]]
       
        else:
            raise Exception('Cannot initialise path')
            

        previous_face = set(self.triangles_containing_edges([ind,ind],edge)[0])
        
        adj_faces = self.triangles_containing_edge(list(edge))
        if set(adj_faces[0]) == previous_face:
            current_face = adj_faces[1]
        else:
            current_face = adj_faces[0]

        path_info = {'points': path, 'gradient' : grad, 'edges': np.array([[ind,ind],edge]), 'faces':[current_face]}
        
        return path_info

    def sorted_neigh_vertex_path(self,path, triangles):
        # First check whether the last point was a node
        if path['edges'][-1][0] == path['edges'][-1][1]:
            center = path['edges'][-1][0]
        else:
            center = -1

        prev_point_is_vertex = path['edges'][-2][0]==path['edges'][-2][1]
        n = []
        start_face = path['faces'][-1]
        i = np.where((triangles[:,:,np.newaxis] == start_face.reshape(1,1,-1)).sum(-1).sum(-1) == 3)[0]
        
        tri = triangles[i[0]]
        if prev_point_is_vertex:
            n.append(path['edges'][-2][0])
        else:
            tri = [v for v in tri if (v != center)]
            n.append(tri[0])
        
        triangles = np.delete(triangles,i,0)
        
        if center == -1:
            i = np.where((triangles == np.array(n[-1])).sum(-1) == 1)
        else:
            
            i = np.where((triangles[:,:,np.newaxis] == np.array([[[center,n[-1]]]])).sum(-1).sum(-1) == 2)[0]
        
        while len(triangles) > 0:
            tri = triangles[i[0]]
            tri = [v for v in tri if ((v != center) and (v != n[-1]))]            
    
            
            n.append(tri[0])
            triangles = np.delete(triangles,i,0)
            i = np.where((triangles[:,:,np.newaxis] == np.array([[[center,n[-1]]]])).sum(-1).sum(-1) == 2)[0]

        n.append(n[0])
        return n

    def calculate_next_point(self, path, edge, ratio, gradient = False):
        
        #To calculate the new point we will need the law of sines
        #b/sin(beta) = c/sin(gamma) => c = bsin(gamma)/sin(beta)
        

        p1, p2, p3 = path['points'][-1], *self.vertices[edge]
                
        angle = np.arccos(np.dot((p2-p1)/np.linalg.norm(p1-p2),(p3-p1)/np.linalg.norm(p3-p1)) )
        
        angle1 = ratio*angle

        side = np.linalg.norm(p1-p2)
        angle2 = np.pi - angle1 - np.arccos(np.dot((p1-p2)/np.linalg.norm(p1-p2),(p3-p2)/np.linalg.norm(p3-p2)))

        side1 = side*np.sin(angle1)/np.sin(angle2)

        next_point = p2 + (p3-p2)/np.linalg.norm(p3-p2)*side1
        if gradient:
            return next_point, side1
        else:
            return next_point
'''
