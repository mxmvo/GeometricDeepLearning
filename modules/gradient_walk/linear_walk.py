import numpy as np
from modules.ddg import discrete_gradient

class LinearWalk():
    def __init__(self,trimesh):#, value_function):
        # Trimesh should be a trimesh object
        self.manifold = trimesh
        


    def run(self, origin, start_ind, func, max_length):
        grad_tri = -discrete_gradient(self.manifold, func)
        # Initialize the path
        path = {'points':[self.manifold.vertices[start_ind]],'faces':[], 'edges':[{start_ind, start_ind}]}
        path['start_ind'] = start_ind

        # What are the stop faces.
        stop_faces = [set(t) for t in self.manifold.triangles_containing_node(origin)]

        # Find initial gradient, expressed in the 2d nbh
        init_grad = self.manifold.grad_vertex_nbh([start_ind],grad_tri)[start_ind]
        face = self.next_face_vert(start_ind, init_grad)
        path['faces'].append(face)

        while True: 
            
            point = path['points'][-1]
            edge = path['edges'][-1]
            face = path['faces'][-1]
            #print(point)
            if face in stop_faces:
                return path

            next_point, next_edge, next_face = self.next_step(point, edge, face, grad_tri)

            path['points'].append(next_point)
            path['faces'].append(next_face)
            path['edges'].append(next_edge)

            path_length = np.linalg.norm((np.array(path['points'])[1:]-np.array(path['points'])[:-1]), axis = 1).sum()
            if path_length > 2*max_length:
                print('no converge, Origin: {}, From: {}'.format(set.intersection(*stop_faces),start_ind))
                return path


        return path


    def next_step(self, point, edge, face, grad_tri):
        edge_list = list(edge)

        if len(edge_list) == 1:
            grad = self.manifold.grad_vertex_nbh(edge_list,grad_tri)[edge_list[0]]
            next_point, next_edge = self.next_point_vertex(edge_list[0], face, grad)
        
        else:
            face_ind  =self.face_ind(face)
            grad = grad_tri[face_ind]
            next_point, next_edge = self.next_point_edge(point, edge, face, grad)


        l_edge = list(next_edge)
        edge_points = self.manifold.vertices[l_edge]

        # Incase the new point is to close to the end points of the edge
        # we will make the new point that vertex. Otherwise we will get non-singular matrices later on.
        if np.allclose(edge_points[0], next_point):
            next_point = edge_points[0]
            next_edge = {l_edge[0]}
        elif np.allclose(edge_points[1] , next_point):
            next_point = edge_points[1]
            next_edge = {l_edge[1]}
        
        l_edge = list(next_edge)
        if len(l_edge) == 1:
            grad = self.manifold.grad_vertex_nbh(l_edge,grad_tri)[l_edge[0]]
            next_face = self.next_face_vert(l_edge[0], grad)
        else:
            next_face = self.next_face_edge(next_edge, face)


        return next_point, next_edge, next_face

    def next_point_vertex(self,ind, face, grad):
        chart = self.manifold.chart(ind)
        nbh = chart['nbh']
        e = list(face-{ind})
        a1, b1 = self.manifold.decompose_vector(grad,nbh[e[0]] , nbh[e[1]])


        if a1<0:
            a, b = 0, 1
        elif b1 < 0:
            a, b = 1, 0
        else:
            a, b = a1/(a1+b1), b1/(a1+b1)
        e1, e2 = self.manifold.vertices[e]
        center = self.manifold.vertices[ind]

        next_point = a*(e1-center)+b*(e2-center)+center


        return next_point, set(e)

    def next_point_edge(self,point, edge, face, grad):
        horizon_ind = (face-edge).pop()
        #print(edge, face, point)
        edge_list = list(edge)
        e1, e2 = self.manifold.vertices[edge_list]
             
        h = self.manifold.vertices[horizon_ind]
        
        a1, b1 = self.manifold.decompose_vector_3D(grad,  e1-point, h-point)
        a2, b2 = self.manifold.decompose_vector_3D(grad,  e2-point, h-point)

        
        if (a1 > 0) and (b1 > 0):
            a, b = a1/(a1+b1), b1/(a1+b1)
            next_point = a*(e1-point)+b*(h-point) + point
            next_edge = {edge_list[0],horizon_ind}
        elif (a2>0) and (b2 > 0):
            a, b = a2/(a2+b2), b2/(a2+b2)
            next_point = a*(e2-point)+b*(h-point) + point
            next_edge = {edge_list[1],horizon_ind}
        else:
            if np.dot(e1-point, grad)>0:
                next_point = e1
                next_edge = {edge_list[0]}
            
            else:
                next_point = e2
                next_edge = {edge_list[1]}
            

        return next_point, next_edge
        


    def next_face_edge(self,edge, prev_face):
        adj_tri = self.manifold.triangles_containing_edge(list(edge))
        
        # On the boundary we only have one triangle
        if len(adj_tri) == 1:
            return set(adj_tri[0])

        # Otherwise.
        for t in adj_tri:
            if set(t) != prev_face:
                return set(t)
            
    def face_ind(self,face):
        f_l = list(face)
        n0 = np.where(self.manifold.triangles == f_l[0])[0]
        n1 = np.where(self.manifold.triangles == f_l[1])[0]
        n2 = np.where(self.manifold.triangles == f_l[2])[0]
        
        return set.intersection(*[set(n0), set(n1), set(n2)]).pop()

    def next_face_vert(self, ind, grad):
        chart = self.manifold.chart(ind)
        nbh, nbh_ind = chart['nbh'], chart['sort_ind']

        edge = self.manifold.which_edge(grad,nbh,nbh_ind)

        if edge == None:
            b1 = nbh[nbh_ind[0]]
            if np.dot(b1, grad)>0:
                edge = nbh_ind[:2]
            else:
                edge = nbh_ind[-2:]

        face = set(edge)
        face.add(ind)
        return face