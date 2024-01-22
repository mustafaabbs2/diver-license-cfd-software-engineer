import numpy as np
import pyvista as pv
import nemo as n
import post_module as pm


pointcloud = n.Particles.load(n.File("D:\\Mustafa\\Simulations\\short_simulation\\fluid_6800.pvtp")).data
# print(pointcloud.points)
# print(pointcloud.attributes["density"] )

def visualize_particles():
    density = pointcloud.attributes["density"]
    pdata = pv.PolyData(pointcloud.points)
    sphere = pv.Sphere(radius=0.02, phi_resolution=10, theta_resolution=10)
    pc = pdata.glyph(scale=False, geom=sphere, orient=False, rng=[np.min(density), np.max(density)])
    pc.plot()


def create_structured_volume():
    x_resolution = 10
    y_resolution = 10

    x = np.linspace(0.07, 0.17, x_resolution)
    y = np.linspace(0.07, 0.17, y_resolution)
    z = np.arange(0, 0.02, 0.01)

    xx, yy, zz = np.meshgrid(x, y, z)

    points = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

    grid = pv.StructuredGrid(xx, yy, zz)

    # Plot the grid
    plotter = pv.Plotter()
    plotter.add_mesh(grid, color="white", show_edges=True)
    plotter.show()

    return grid

def create_volume_from_rectilinear():
    x_resolution = 10
    y_resolution = 10

    x = np.linspace(0.07, 0.17, x_resolution)
    y = np.linspace(0.07, 0.17, y_resolution)
    z = np.zeros(1)  # 2D box, so z dimension has only one value

    grid = pv.RectilinearGrid()
    grid.x = x
    grid.y = y
    grid.z = z

    grid.plot("show_edges=True")
    grid = grid.cast_to_structured_grid()
    return grid

def write_volume_to_vtk(grid, filename):
    grid.plot("show_edges=True")
    mesh = grid.cast_to_unstructured_grid()
    mesh.save(filename)

def write_volume_to_stl(grid, filename):
    grid.plot("show_edges=True")
    mesh = grid.cast_to_unstructured_grid()
    mesh = mesh.extract_surface()
    mesh.save(filename)


def read_into_nemo(file): 
     """read into nemo, could be a .vtu or a .stl

     Args:
        file: filename.
     """
     read_volume(file)
     mesh =  n.Mesh.load(n.File(file))
     segmentcloud = mesh.data
     areas =  mesh.get_surface_area_per_segment()
     normals = segmentcloud.get_segment_normals()
     return areas, normals, mesh

def read_volume(file):
     mesh_display = pv.read(file)
     mesh_display.plot(show_edges=True)


def riemann_sum(field, areas, normals):
    #Riemann sum of a unit vector
    dot_product = np.sum(normals, axis=1) #horizontally sum across normals -> 1
    riemann_sum = np.sum(dot_product * areas)

    print('Riemann sum is:')
    print(riemann_sum)



# Step 1: Display particles
# visualize_particles()
    
# Step 2: Write CV grid to STL
filename = "output_vol.stl"
grid = create_structured_volume()
write_volume_to_stl(grid, filename)
read_volume(filename)

#Step 3: Read back into nemo
# areas, normals, mesh = read_into_nemo(filename)




#Do a closest neighbour search of particles within each grid element --> I'm not entirely sure how to do this right now 