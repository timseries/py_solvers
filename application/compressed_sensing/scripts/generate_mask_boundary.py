#script to generate a mask boundary 
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import dilation,disk

#input mask path
mask_path = '/home/tim/repos/py_solvers/application/data/velocity_imaging/reducing_times_benchmark_spatial_mask.npz'
mask = np.load(mask_path)['arr_0']

disk_struct_el = disk(2)
imones = np.ones(mask.shape, dtype='bool')
boundary_mask = mask - (imones - dilation(imones - mask, disk_struct_el))

plt.imshow(mask)
plt.figure()
plt.imshow(boundary_mask)
plt.figure()
plt.imshow(boundary_mask*mask)

#save
np.savez_compressed('/home/tim/repos/py_solvers/application/data/velocity_imaging/reducing_times_benchmark_spatial_mask_boundary.npz',boundary_mask)