import numpy as np

eps = np.finfo(np.float64).eps
eps32 = np.finfo(np.float32).eps

class QuadraticWalk():
    def __init__(self,trimesh):#, value_function):
        # Trimesh should be a trimesh object
        self.manifold = trimesh
        

    def run(self, origin, ind, func, max_length, step_size = 0.0001):
        # Initialise a path
        path = {}
        path['points'] = np.array([self.manifold.vertices[ind]])
        path['faces'] = [set(self.manifold.triangles_containing_node(ind)[0])]
        path['start_ind'] = ind
        path_length = 0

        stop_faces = [set(t) for t in self.manifold.triangles_containing_node(origin)]

        bary_mat = np.linalg.inv(np.transpose(self.manifold.vertices[self.manifold.triangles], (0,2,1)))

        
        while (path_length < 10*max_length):# and (len(path['points']) < 40000):

            # Get the knowns
            current_point = path['points'][-1]
            current_face = set(path['faces'][-1])
            if set(current_face) in stop_faces:
                
                break
            
            center_vertex = self.find_center_vertex(current_point,current_face, bary_mat)
            
           
            # Make the neighborhood
            # Input: the center_vertex and the current_point
            # Return dict from indices+['current'] to R2
            chart = self.manifold.chart(center_vertex)
            nbh_hood, nbh_ind = chart['nbh'], chart['sort_ind']

            # Make grad_hass approximation
            # Input: {ind: val, ...} and {ind: [x,y],.....}
            # Output: [grad:hess] matrix

            grad_hess_approx = self.manifold.grad_hess_approx(func, nbh_hood, center_vertex)
            
            # Use Heun algorithm untill we are in different face
            # Input: function approx, neighborhood, current_edge
            # current edge is needed for a stopping criterion
            # output: [multiple points], [edge], [face]

            pnts_2D, edges = self.run_heun(current_point, grad_hess_approx, nbh_hood, nbh_ind, current_face, bary_mat, step_size_init = step_size)

            #pnts_2D, edges = self.walk_linear(current_point, grad_hess_approx, nbh_hood, nbh_ind, current_face, bary_mat)
            
            pnts_3D = self.manifold.chart_to_mesh(pnts_2D, nbh_hood, nbh_ind, center_vertex)
            
            if len(pnts_3D) > 0:
                next_face = edges[-1]
                next_face.add(center_vertex)

                path['points'] = np.vstack([path['points'], pnts_3D])
                path['faces'].append(next_face)

            
            path_length = np.linalg.norm((path['points'][1:]-path['points'][:-1]), axis = 1).sum()
            if path_length > 10*max_length:
                print('Length:', path['faces'][-1], center_vertex)
                print(path)
        # Check endpoint in neighborhood, by bary
        return path

    def find_center_vertex(self, pnt, face, bary_mat):
        f = list(face)
        
        f_ind = self.manifold.face_ind(f)
        f = self.manifold.triangles[f_ind]

        a, b, c = list(np.dot(bary_mat[f_ind],pnt))

        if c < 1/3:
            if a>b:
                return f[0]
            else:
                return f[1]
        else:
            if a > b:
                if c > a:
                    return f[2]
                else:
                    return f[0]
            else:
                if c > b:
                    return f[2]
                else:
                    return f[1]


    def run_heun(self, x0, grad_hessian, nbh_hood, nbh_ind, current_face, bary_mat, step_size_init):
        '''
        y'(t) = f(t,y(t))
        
        y_pred = y_t + hy'(t)
        y_{t+1} = y_t + h/2( f(t,y_t)+ f(t+1,y_pred) )
        '''
        points_2D = [] 
        edges = []
        def calc_gradient(x, grad_hessian= grad_hessian, nbh_ind = nbh_ind):
            '''
            f(x) = [Df:Df^2]*[x,y, .5x^2, xy, .5y^2].T 
            grad(f) = [[Df:Df^2]*[1,0,x,y,0],[Df:Df^2]*[0,1,0,x,y]
            '''
            
            if len(grad_hessian) > 2:
                x,y = list(x)
                coord = np.array([[1,0],[0,1],[x,0],[y,x],[0,y]])
                p =  -np.dot(grad_hessian,coord)
            else:
                p = -grad_hessian

            # If we are on the boundary, then we walk along the edge.
            if nbh_ind[0] != nbh_ind[-1]:
                p[1] = max(eps, p[1])
            
            return p/np.linalg.norm(p)

        x0 = self.manifold.mesh_to_chart(x0, current_face, nbh_hood)
        current_edge = current_face.intersection(set(nbh_ind))
        center_vertex = list(current_face.difference(current_edge))[0]
        
        step_size = step_size_init
        while True:
            grad0 = calc_gradient(x0)
            x_tilde = x0 + step_size*grad0
            grad1 = calc_gradient(x_tilde)

            x1 = x0 + step_size*(grad0+grad1)/2
            edge = self.manifold.which_edge(x1, nbh_hood, nbh_ind)
            
            
            if edge == None:
                # We have moved outside the nbh
                # Decrease stepsize if we haven't found new points.
                if len(points_2D) == 0:
                    step_size = step_size/2
                    if step_size < eps:
                        # Stepsize is too small
                        raise ValueError('Stepsize is too small neighbor hood is too small to handle')
                else:
                    return points_2D, edges
            else:
                points_2D.append(x1)
                edges.append(edge)
                

                if edge != current_edge:
                    # We are in a different triangle return points
                    return points_2D, edges
                else:
                    # See if we should still use this center vertex
                    e= list(current_edge)

                    x0_3d = self.manifold.chart_to_mesh([x0], nbh_hood,nbh_ind, center_vertex)[0]
                    c_v = self.find_center_vertex(x0_3d,{center_vertex, e[0],e[1]},bary_mat)

                    
                    if c_v != center_vertex:
                        return points_2D, edges
                    else:
                        if all(x0 == x1):
                            # We are in a minimum.
                            return points_2D, edges
                        x0 = x1