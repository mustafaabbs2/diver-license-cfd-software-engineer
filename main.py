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


def write_to_stl():
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
    mesh = grid.cast_to_structured_grid()
    mesh = mesh.extract_surface()
    mesh.save("output.stl")

def read_into_nemo(): #note I could only read back an STL, so it's going to be triangulated..
     mesh_display = pv.read("output.stl")
     mesh_display.plot()

     mesh =  n.Mesh.load(n.File("output.stl"))
     segmentcloud = mesh.data
     areas =  mesh.get_surface_area_per_segment()
     normals = segmentcloud.get_segment_normals()

    #  print(segmentcloud.points)
    #  print(segmentcloud.segments)
    #  print(segmentcloud.point_attributes) #-->empty dict, how to proceed?
    #  print(segmentcloud.point_attributes["face_normal"])# KeyError: 'face_normal'
    #  print(segmentcloud.point_attributes["area"])# KeyError: 'area'
     
     #Riemann sum of a unit vector
     dot_product = np.sum(normals, axis=1) #horizontally sum across normals -> 1
     riemann_sum = np.sum(dot_product * areas)
    
     print('Riemann sum is:')
     print(mesh.get_surface_area())
     print(riemann_sum)





# Step 1: Display particles
visualize_particles()
# Step 2: Write CV grid to STL
write_to_stl()

#Read back into nemo
#Do a closest neighbour search of particles within each grid element --> I'm not entirely sure how to do this right now 
read_into_nemo()
