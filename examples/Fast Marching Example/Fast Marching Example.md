

```python
import sys
sys.path.append("..")

%load_ext autoreload
%autoreload 2
```


```python
import numpy as np

from modules.fast_marching_method import FMM
from modules.trimesh import trimesh

from plyfile import PlyData, PlyElement

import vtki
from vtki import PolyData
```

# Fast Marching Method

This post is dedicated to the fast marching method as described in [Computing geodesic paths on manifolds](http://www.pnas.org/cgi/doi/10.1073/pnas.95.15.8431). 

It uses a sweeping method, meaning that it is only concerned with the neighborset of known (already calculated) distances. 

In short:
- From the sweep-area pick the vertex with the smallest distance. Remove from the sweep area and put them in the alive set. 
- Add potential new points to the sweep-area.
- Update the values of the points in the sweep area. 

Continue untill all the points are calculated or when a desired distance is reached.

The approach taken in the paper for the update rule is based on the triangulation. 
![Image](./images/fmm.png?raw=true)

The distance to point $A$ and $B$ are known and are such that $d(B)\geq d(A)$. The goal is to find a 'distance' plane that has the distance values of $A$ and $B$. Right now there are infinitely many planes that obey this rule. With the following observations we can restrict the solution space.

Because of the sweeping method, we can assume the distance to $C$ is bigger than $d(B)$. Thus there will be a point $D$, on line $AC$ that will atain the value $d(B)$. Note that the gradient of the distance function should be $1$, as you increase one unit if you move one unit. Using this fact, we should know how much the plane should increase from line $BD$ to $C$.

Using basic trigonometric functions we can express the desired quantity as the smallest root of the 2nd order polynomial. These detail will be omitted here and can be found in the original article.

---

Another approach that gives the same results is as follows. It is still assumed that $d(A)$ and $d(B)$ are known. Again the norm of the gradient of the distance function should be $1$. The gradient on a face $f$, can be described as

\begin{align*}
(\nabla d)_f = \frac{1}{2A_f} \sum_i d_i \left(N \times e_i\right)
\end{align*}
Here $A_f$ is the area of the face, $N$ the normal vector of the triangle, $e_i$ the edges ordered counter clockwise and finally $d_i$ the opposing value. 

Using standard cross product rules, the above requirements also reduces to a 2nd order polynomial equation. 

\begin{align*}
1 = \sum_{ij} d_i d_j \langle e_i, e_j \rangle
\end{align*}

---

For both approaches the point should be updated from within the triangle. Meaning that the negative gradient should be pointing in the triangle. Since on a 3D-mesh one cannot know that the shortest path can be reached from outside the triangle, e.g. boundaries. If this happens we set the value as the length of the edge to C plus the distance of the known vertices of that edge. 

**note:** The code could probably use some performance boosts

# Below some functions used for loading data and plotting


```python
def read_ply(f_name):
    # Read the vertices and triangles from a ply file
    data = PlyData.read(f_name)
    
    data_vert = np.vstack([list(vertex) for vertex in data['vertex'].data])
    data_tri = np.vstack(data['face'].data['vertex_indices'])
    return data_vert, data_tri

color_map = 'prism'
color_point = 'black'
def vtki_plot(mesh,ind, color, cmap = color_map, arrows_tri = [], paths = [], text = '', inline = False, show_edges = True):
    # Plot functions on the mesh, together with arrows
    polydata = PolyData(mesh.vertices, np.c_[[[3]]*len(mesh.triangles),mesh.triangles])
    if inline:
        plotter = vtki.Plotter(notebook = True)
    else:
        plotter = vtki.BackgroundPlotter()

    plotter.add_mesh(polydata, scalars = color, cmap = cmap, show_edges = show_edges)
    plotter.add_text(text)
    
    if len(paths) > 0 :
            for pa in paths:
                p = np.array(pa['points'])
                if len(p) > 0:
                    plotter.add_lines(p, width = 50, color = 'white')
    
    if len(ind)> 0:
        plotter.add_points(mesh.vertices[ind], point_size = 20, color = 'black')

    if len(arrows_tri) > 1:
        cent = mesh.vertices[mesh.triangles].mean(1)
        plotter.add_arrows(cent, arrows_tri, mag  = .01, color = 'g')
    plotter.view_xy()
    plotter.show()    

```

# Example: Standford Bunny



```python
bunny_vert, bunny_tri = read_ply('../test_data/bunny_36k.ply')
bunny = trimesh(bunny_vert,bunny_tri)
```


```python
# Initialise the algorithm
march = FMM(bunny)
```


```python
ind0 = np.argmax(bunny_vert[:,0])
ind1 = np.argmax(bunny_vert[:,1])
ind2 = np.argmax(bunny_vert[:,2])
ind3 = np.argmin(bunny_vert[:,0])
ind4 = np.argmin(bunny_vert[:,2])
```


```python
%%time
# Run the algorithm from a index untill al the points in the sweep area have values bigger then d_max
d_max = .02
d  = march.run([ind0, ind1, ind2,ind3,ind4], d_max)
```

    CPU times: user 9.44 s, sys: 47.5 ms, total: 9.48 s
    Wall time: 9.89 s



```python
d[d == np.inf] = 0
```


```python
#vtki_plot(bunny,[ind0, ind1, ind2, ind3, ind4], d, cmap = 'jet', show_edges = False)
```


```python
plotter = vtki.Plotter(shape=(2, 2))

polydata = PolyData(bunny_vert, np.c_[[[3]]*len(bunny_tri),bunny_tri])
color_map = 'jet'

plotter.subplot(0,0)
plotter.add_text('View 1', position=None, font_size=30)
plotter.add_mesh(polydata, scalars = d, cmap = color_map)
plotter.view_vector([-1,0,1], viewup = [0,1,0])


plotter.subplot(0, 1)
plotter.add_text('View 2', font_size=30)
plotter.add_mesh(polydata, scalars = d, cmap = color_map)
plotter.view_vector([-1,0,-1], viewup = [0,1,0])

plotter.subplot(1, 0)
plotter.add_text('View 3', font_size=30)
plotter.add_mesh(polydata, scalars = d, cmap =color_map)
plotter.view_vector([1,0,-1], viewup = [0,1,0])

plotter.subplot(1, 1)
plotter.add_text('View 4', font_size=30)
plotter.add_mesh(polydata, scalars = d, cmap = color_map)
plotter.view_vector([1,0,1], viewup = [0,1,0])


plotter.show()
```


![png](output_11_0.png)



```python

```
