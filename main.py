import numpy as np

import nemo as n

import post_module as pm

# mesh: n.Mesh = n.cylinder(radius=1, height=1, axis=n.Vector(0, 1, 0))
# print("")
# print("mesh connectivity:")
# print(mesh.data.segments[:3, :])
# print("...")
# print(mesh.data.segments[-3:, :])


# print("")
# print("add:")
# print(pm.add(1, 3))
# print("")

# print("modify_array:")
# a = np.arange(100_000_000).astype(np.float64)
# print(a)
# pm.modify_array(a)
# print(a)
# print("")

# print("make_new_array:")
# b = pm.make_new_array(a, 4.0)
# print(b)
# print("")

# print("create_data:")
# for k, v in pm.create_data().items():
#     print(k, v, type(v))
# print("")


pointcloud = n.Particles.load(n.File("D:\\Mustafa\\Simulations\\short_simulation\\fluid_6800.pvtp")).data
print(pointcloud.points)
print(pointcloud.attributes["density"] )

