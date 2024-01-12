import numpy as np

import nemo as n

import post_module as pm

mesh: n.Mesh = n.cylinder(radius=1, height=1, axis=n.Vector(0, 1, 0))

print(pm.add(1, 3))

a = np.arange(100_000_000).astype(np.float64)

pm.modify_array(a)
b = pm.make_new_array(a, 4.0)

print(a)
print(b)
