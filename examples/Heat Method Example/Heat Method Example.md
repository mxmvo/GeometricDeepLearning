

```python
import sys
import vtki
import numpy as np

from plyfile import PlyData, PlyElement
from vtki import PolyData
from tqdm import tqdm

sys.path.append("../..")
from modules.heat_method import heat_method

```

# Table of content:
- [Heat method for computing geodesic distance](#Intro): Explaination of the method used
- [Examples](#Examples): Some examples where the heat method has been applied
- [Issues](#Issues): Where have I experienced issues with the method

<a name="Intro"></a>
# Heat method for computing geodesic distance.

The heat method is based on the article [Geodesics in Heat](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/).

The algorithm uses discretisation of certain mathematical operators. This will eventually results into solving linear equations, which can be done using standard packages. On of the advantages of this method over Fast Marching Method is that the algoritm can already precompute certain matrices and factorise those, making the algorithm efficient when the goal is to calculate the distance from multiple source points. 

Another advantage is that the triangulation does not have as big of an impact, as it would have in the FMM case. In the FMM when a triangle is not acute then there is an unfolding principle. For more information about the FMM see [Computing geodesic paths on manifolds](http://www.pnas.org/cgi/doi/10.1073/pnas.95.15.8431)

## The Method
The Heat Method consists of three main steps:
- I. Integrate the heat flow $\dot{u} = \Delta u$ for some fixed time $t$
- II. Evaluate the vector field $X = −\nabla u/|\nabla u|$.
- III. Solve the Poisson equation $\Delta \phi = \nabla \cdot X$.

I will not argue for the validity of the method, if you are interested I refer to the article. In the article they set $t =m*h^2$, here $h$ is the mean distance between adjacent vertices and $m$ a constant bigger than $0$. Theoretically $t$ should be as small as possible, numerically $m=1$ works fine according to the authors. If the mesh is not uniform, they suggest a $h$ to be the max distance between adjacent vertices. 

## The Discretisation

The discretisation of the above described steps is given below, if you are interested in what this actually means look at the next section. If you want to see examples scroll further down.
\begin{align*}
\left(A-t L\right) u & =u_{0}& \text{I}\\
(\nabla u)_f& =\frac{1}{2 A_{f}} \sum_{i} u_{i}\left(N \times e_{i}\right) & \text{II}\\
(\nabla \cdot X)_v& =\frac{1}{2} \sum_{j} \cot \theta_{1}\left(e_{1} \cdot X_{j}\right)+\cot \theta_{2}\left(e_{2} \cdot X_{j}\right) & \text{II}\\
L \phi & =\nabla X & \text{III}
\end{align*}

These discritisation assume that the on the boundary the gradient of the function is zero. 

### Details about the Discretisation

For discretisation of step I. An Euler backward in time $ \dot{u_t} = \frac{u_t-u_0}{t}$ here $u_t$ is the heat after a certain time, and $u_0$ is the initial heat, which in our case is a certain point thus a vector with $1$ at the desired vertex. The Discrete Laplace Beltrami operator can be rewritten as $\Delta = A^{-1}L_c$. Here $A$ is a diagonal matrix with on position $ii$ the vertex area of $i$. This is commonly taken to be the a third of the sum of the area's containing $i$, this is a bary centric area. Another approach could be the circumcentric area, this takes the area of the dual cell, dual cells are better described in  [Discrete Differential Forms for Computational Modeling](http://www.geometry.caltech.edu/pubs/DKT05.pdf), this reference is a nice introduction to a way of discritizing certain mathematical operators, a somewhat different approach than the finite element methods.

The matrix $L$ is based on the cotan formula, and is 

\begin{align*}
  (L)_{ij} = \begin{cases}
    (\cot(\alpha_{ij}) + \cot(\beta_{ij}))/2 & ij \in int (E) \\
    \cot(\alpha_{ij})/2 & ij \in \partial E \\
    -\sum_{k\neq i} L_{ik} & i =j \\
    0 & else
  \end{cases}
\end{align*}
where $\alpha_{ij}$ and $\beta_{ij}$ are the angles opposite the edge $ij$. 

With this notation we can discritisize step I
\begin{align*}
\left(A-t L\right) u & =u_{0}& \text{I}\\
\end{align*}

Note that the matrix, $A-tL$, can precomputed and prefactorized since it does not depend on the source point. 

The discretisation of the gradient of a function that is defined on the vertices.
\begin{align*}
(\nabla u)_f & =\frac{1}{2 A_{f}} \sum_{i} u_{i}\left(N \times e_{i}\right) & \text{II}\\
\end{align*}
In this formula $\nabla u$ is computed for each triangle/face, $f$, and $A_f$ is the area of that triangle. Further $N$ denotes the normal vector of that triangle and $e_i$ the respective edges in counter clockwise order. 

The vector field $X$ is defined for each triangle, the divergence of this vector field is again a function on the vertices. The discretisation of the divergense is, 
\begin{align*}
(\nabla \cdot X)_v& =\frac{1}{2} \sum_{j} \cot \theta_{1}\left(e_{1} \cdot X_{j}\right)+\cot \theta_{2}\left(e_{2} \cdot X_{j}\right) & \text{II}\\
\end{align*}
The summation is over triangles that contain $v$, $e_1$ and $e_2$ are the edges of that triangle originating from $v$. The angle $\theta_1$ is opposite the edge $e_1$, similarly for angle $\theta_2$. Also in the discretisations of step II we can precompute certain quantities.

As a final step we need to solve the final system of equations. 
\begin{align*}
L \phi & =\nabla \cdot X & \text{III}
\end{align*}
Also here we can factorise $L$.


A carefull reader migth say that in step I, there is a multiplication with the area matrix missing on the righthand side. This is true, expect that if we have a diagonal matrix, this reduces to a scalar multiplication (for a single source point). This will then cancel in the normalisation of the vector field, thus we leave it out. 

In step III there seems to a again a multiplication missing of on the left hand side with $A^{-1}$. The reason we can omit this is that the actual discritisation of the divergence has a extra factor that is exactly $A^{-1}$, thus they will also cancel each other. See [Discrete Exterior Calculus](http://arxiv.org/abs/math/0508341) and [Discrete Differential Geometry: An Applied Introduction](https://www.cs.cmu.edu/~kmcrane/Projects/DGPDEC/) for formal derivations of the discretisations. 

---

## Some auxiliary functions

A function to read a ply file and return numpy arrays, and a function that uses the vtki plotter to show the calculated distance function. The heat method will only be based on the numpy arrays, this way is can easily fit in a different project. Note that for the plotter is the plot is inline we cannot manipulate it, set to false we can rotate and zoom in in a different window. 


```python
def read_ply(f_name):
    # Read the vertices and triangles from a ply file
    data = PlyData.read(f_name)
    
    data_vert = np.vstack([list(vertex) for vertex in data['vertex'].data])
    data_tri = np.vstack(data['face'].data['vertex_indices'])
    return data_vert, data_tri

color_map = 'prism'
color_point = 'black'
def vtki_plot(vert, tri ,ind, color, cmap = color_map, arrows_tri = [], text = '', inline = True, show_edges = True):
    # Plot functions on the mesh, together with arrows
    polydata = PolyData(vert, np.c_[[[3]]*len(tri),tri])
    if inline:
        plotter = vtki.Plotter(notebook = True)
    else:
        plotter = vtki.BackgroundPlotter()

    plotter.add_mesh(polydata, scalars = color, cmap = cmap, show_edges = show_edges)
    plotter.add_text(text)
    if len(ind)> 0:
        plotter.add_points(vert[ind], point_size = 10, color = 'red')

    if len(arrows_tri) > 1:
        cent = vert[tri].mean(1)
        plotter.add_arrows(cent, arrows_tri, mag  = .01, color = 'g')
    plotter.view_xy()
    plotter.show()    

```

<a name="Examples"></a>
# Examples
## Bunny

Lets start of with the standford bunny, this is a smaller version with 14290 vertices and 28576 triangles.
Initially $t$ is the mean distance squared between adjacent nodes. Later on we will add a scalar to $t$, and see how the distance changes. 


```python
bunny_vert, bunny_tri = read_ply('../test_data/bunny_14k.ply')

# Instansiating the algorithm
# Here some preprocessing steps are being taken
heat_bunny = heat_method(bunny_vert, bunny_tri, t_mode = 'mean')
```


```python
# Finding the index that on the leg of the bunny
ind_bunny = np.argmax(bunny_vert[:,2])
# Run the algorithm for a certain index
d_bunny, X_bunny = heat_bunny.run(ind_bunny)
d_bunny[ind_bunny]
```




    0.0




```python
plotter = vtki.Plotter(shape=(2, 2))

polydata = PolyData(bunny_vert, np.c_[[[3]]*len(bunny_tri),bunny_tri])

plotter.subplot(0,0)
plotter.add_text('Render Window 0', position=None, font_size=30)
plotter.add_mesh(polydata, scalars = d_bunny, cmap = color_map)
plotter.add_points(bunny_vert[ind_bunny], point_size = 10, color = color_point)
plotter.view_vector([0,0,1], viewup = [0,1,0])

plotter.subplot(0, 1)
plotter.add_text('Render Window 1', font_size=30)
plotter.add_mesh(polydata, scalars = d_bunny, cmap = color_map)
plotter.add_points(bunny_vert[ind_bunny], point_size = 10, color = color_point)
plotter.view_vector([-1,0,0], viewup = [0,1,0])

plotter.subplot(1, 0)
plotter.add_text('Render Window 2', font_size=30)
plotter.add_mesh(polydata, scalars = d_bunny, cmap =color_map)
plotter.add_points(bunny_vert[ind_bunny], point_size = 10, color = color_point)
plotter.view_vector([0,0,-1], viewup = [0,1,0])

plotter.subplot(1, 1)
plotter.add_text('Render Window 3', font_size=30)
plotter.add_mesh(polydata, scalars = d_bunny, cmap = color_map)
plotter.add_points(bunny_vert[ind_bunny], point_size = 10, color = color_point)
plotter.view_vector([1,0,0], viewup = [0,1,0])



plotter.show()
```


![png](output_7_0.png)



```python
# Running with a m scalar forces us to factorise a matrix, this is done and saved for further querries.
d_bunny_01, X_bunny_01 = heat_bunny.run(ind_bunny, m = .1)
d_bunny_10, X_bunny_10 = heat_bunny.run(ind_bunny, m = 10)
d_bunny_100, X_bunny_100 = heat_bunny.run(ind_bunny, m = 100)

d_bunny_01[ind_bunny], d_bunny_10[ind_bunny], d_bunny_100[ind_bunny]
```




    (0.0, 0.0, 0.0)




```python
# The plotter behaves a bit weird, I can not get the right images in the desired frames
# But this seems to work oke
plotter = vtki.Plotter(shape=(2 ,4))

polydata = PolyData(bunny_vert, np.c_[[[3]]*len(bunny_tri),bunny_tri])

times = [.1,1,10,100]
dist = [d_bunny_01, d_bunny, d_bunny_10, d_bunny_100]
view = [[0,0,1], [0,0,-1]]

for j in range(2):
    for i in range(4):
        plotter.subplot(i,j)
        plotter.add_text('m: {}'.format(times[i]), position=None, font_size=30)
        plotter.add_mesh(polydata, scalars = dist[i], cmap = color_map)
        plotter.add_points(bunny_vert[ind_bunny], point_size = 10, color = color_point)
        plotter.view_vector(view[j], viewup = [0,1,0])


plotter.show()
```


![png](output_9_0.png)


Increasing $m$ results in a longer flow, making the distance somewhat smoother, this will then ofcourse not be the same distance as before. Theoretically you should have $t \to 0$. Visually it seems that equal distances still get the same number, i.e. is looks like a distance function, but not one derived from euclidian innerproduct. 

----

## Sphere

Below a exposition how the method works using different triangulations. For the sphere we regard the iso and the uv triangulation.

### North Pole


```python
# The iso-triangulated sphere
sphere_iso_vert, sphere_iso_tri = read_ply('../test_data/sphere_iso_10k.ply')
heat_iso = heat_method(sphere_iso_vert, sphere_iso_tri)
```


```python
# Calculating
ind_sph_iso_north = np.argmax(sphere_iso_vert[:,2])
d_iso_north, _ = heat_iso.run(ind_sph_iso_north)
d_iso_north[ind_sph_iso_north]
```




    0.0




```python
# The uv-triangulated sphere
sphere_uv_vert, sphere_uv_tri = read_ply('../test_data/sphere_uv_10k.ply')
heat_uv = heat_method(sphere_uv_vert, sphere_uv_tri)
```


```python
# Calculating
ind_sph_uv_north = np.argmax(sphere_uv_vert[:,2])
d_uv_north, _ = heat_uv.run(ind_sph_uv_north)
d_uv_north[ind_sph_uv_north]
```




    0.0




```python
poly_uv = PolyData(sphere_uv_vert, np.c_[[[3]]*len(sphere_uv_tri),sphere_uv_tri])
poly_iso = PolyData(sphere_iso_vert, np.c_[[[3]]*len(sphere_iso_tri),sphere_iso_tri])

plotter = vtki.Plotter(shape=(1, 2))

plotter.subplot(0,0)
plotter.add_text('UV: north pole', position=None, font_size=30)
plotter.add_mesh(poly_uv, scalars = d_uv_north, cmap = color_map)#, show_edges = True)
plotter.add_points(sphere_uv_vert[ind_sph_uv_north], point_size = 10, color = color_point)
plotter.view_vector([1,0,1], viewup = [0,0,1])

plotter.subplot(0, 1)
plotter.add_text('Iso: north pole', position=None, font_size=30)
plotter.add_mesh(poly_iso, scalars = d_iso_north, cmap = color_map)#, show_edges = True)
plotter.add_points(sphere_iso_vert[ind_sph_iso_north], point_size = 10, color = color_point)
plotter.view_vector([1,0,1], viewup = [0,0,1])

plotter.show()
```


![png](output_16_0.png)


The distance lines are fairly similar, the difference of the triagnulation is noticable since the uv-decompostion has nice latitude lines, which the iso-decomposition doesn't have. 

### Point on the equator


```python
ind_sph_uv_equator = np.argmax(sphere_uv_vert[:,0])
d_uv_equator, _ = heat_uv.run(ind_sph_uv_equator)


ind_sph_iso_equator = np.argmax(sphere_iso_vert[:,0])
d_iso_equator, _ = heat_iso.run(ind_sph_iso_equator)

d_uv_equator[ind_sph_uv_equator], d_iso_equator[ind_sph_iso_equator]
```




    (0.0, 0.0)




```python
plotter = vtki.Plotter(shape=(1, 2))

plotter.subplot(0,0)
plotter.add_text('UV: equator', position=None, font_size=30)
plotter.add_mesh(poly_uv, scalars = d_uv_equator, cmap = 'prism')#, show_edges = True)
plotter.add_points(sphere_uv_vert[ind_sph_uv_equator], point_size = 10, color = 'red')
plotter.view_vector([1,0,.5], viewup = [0,1,0])

plotter.subplot(0, 1)
plotter.add_text('Iso: equator', position=None, font_size=30)
plotter.add_mesh(poly_iso, scalars = d_iso_equator, cmap = 'prism')#, show_edges = True)
plotter.add_points(sphere_iso_vert[ind_sph_iso_equator], point_size = 10, color = 'red')
plotter.view_vector([1,0,.5], viewup = [0,1,0])

plotter.show()
```


![png](output_19_0.png)


Here the iso-triangulation works with similar results as for the northpole, while the uv decomposition is a bit more fuzzy, and the distance does not seem totally cricular. 

## Origin has non-zero distance

This is a more general mesh of a human body from the [FAUST dataset](http://files.is.tue.mpg.de/black/papers/FAUST2014.pdf)

This mesh is still relatively clean, calculating the distance function there might be a different global minumum instead of the origin point, this could be circumvented using a different $m$ scale. To see this properly one should view the meshes in a different window (here one can zoom and rotate), see the below code for the generation of the images.
<div style="display: block">
<img src="heat_1.png" width="450", style="display: block; float: left;"/><img src="heat_2.png" width="450" style="display: block; float: left;"/>
</div>



The actual origin is in the arm pit. The green arrows are the vector field $X$, i.e. the normalisation of the negative heat flow gradient. The red dot on the heat in the upper right corner is the global minimum (as calculated using $m=1$. For $m=2$ the arrows in the head are pointing towards the previous global minimum, giving a better representation of the distance.


```python
# Reading the body file and initiating the heat method
body_vert, body_tri = read_ply('../test_data/tr_reg_000.ply')
heat_body = heat_method(body_vert, body_tri)
```


```python
# A check for which indices the origin in not the global minimum
# output the indices for which this is the case

#from tqdm import tqdm
#for i in tqdm(range(len(body_vert))):
#    d, X = heat_body.run(i)
#    if d[i] != 0:
#        print(i)
```


```python
# A function to find the neighbors of a index including the point itself
neigh = lambda x: set(body_tri[np.where(body_tri == x)[0]].reshape(-1))

# Origin whicha are note global minimum, computed with the code above
wrong_ind = [1164, 1545, 3286, 4163, 4875]
```


```python
# For each of the indices, output:
# index, min_index, are_neighbors, and dist[index]
dist = []
for i in wrong_ind:
    d, X = heat_body.run(i)
    min_ind = np.argmin(d)
    print(i, min_ind, min_ind in neigh(i), d[i] )
    dist.append([d,X])
```

    1164 1163 True 0.0006047763735645972
    1545 1544 True 0.0008103617268195862
    3286 3311 True 0.0007720704235878451
    4163 177 False 0.05755220581795739
    4875 4162 True 0.0031038005551227155



```python
# Calculate the distance function
d,X = heat_body.run(4163)

# Calculate the distance with a different scale and plot
d2,X2 = heat_body.run(4163, m = 2)
print(d[4163], d2[4163])
```

    0.05755220581795739 0.0



```python
plotter = vtki.Plotter(shape=(1, 2))

# Somehow the reusing the polyData copied the color map, so I had to make two.
poly_body = PolyData(body_vert, np.c_[[[3]]*len(body_tri),body_tri])
poly_body2 = PolyData(body_vert, np.c_[[[3]]*len(body_tri),body_tri])

plotter.subplot(0,0)
plotter.add_text('m: 1', position=None, font_size=30)
plotter.add_mesh(poly_body, scalars = d, cmap = 'prism')
plotter.view_vector([0,0,1], viewup = [0,1,0])

plotter.subplot(0, 1)
plotter.add_text('m: 2', position=None, font_size=30)
plotter.add_mesh(poly_body2, scalars = d2, cmap = 'prism')
plotter.view_vector([0,0,1], viewup = [0,1,0])

plotter.show()
```


![png](output_28_0.png)



```python

```
