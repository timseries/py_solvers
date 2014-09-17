#script to generate a mask boundary 
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import dilation,disk

#input mask path
# mask_path = '/home/tim/repos/py_solvers/application/data/velocity_imaging/2dsine_spatial_mask.npz'
mask_path ='/home/tim/repos/py_solvers/application/data/velocity_imaging/vessels_spatial_mask.npz'
mask = np.load(mask_path)['arr_0']

disk_struct_el = disk(4)
imones = np.ones(mask.shape, dtype='bool')
boundary_mask = mask - (imones - dilation(imones - mask, disk_struct_el))
boundary_mask = np.asarray(boundary_mask,dtype='bool')
not_boundary_mask = np.asarray(mask*(1-boundary_mask),dtype='bool')

plt.imshow(mask)
plt.figure()
plt.imshow(boundary_mask)
plt.figure()
plt.imshow(boundary_mask*mask)
plt.figure()
plt.imshow(not_boundary_mask)

#save boundary mask
# np.savez_compressed('/home/tim/repos/py_solvers/application/data/velocity_imaging/2dsine_spatial_mask_boundary.npz',boundary_mask*mask)

np.savez_compressed('/home/tim/repos/py_solvers/application/data/velocity_imaging/vessels_spatial_mask_boundary.npz',boundary_mask*mask)



