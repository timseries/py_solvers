#script to generate a mask boundary 
import matplotlib.pyplot as plt
import numpy as np
from mpldatacursor import datacursor

#generate a 2-dimensional sinusoid
str_type='qbgn'

periods_x = 3.0
periods_y = 3.0
sz_x = 168.0
sz_y = 168.0

x = np.arange(sz_x)/sz_x*periods_x*2.0*np.pi
y = np.arange(sz_y)/sz_y*periods_y*2.0*np.pi

x,y = np.mgrid[0.1:.1+2.0 * np.pi * periods_x : 1 / sz_x * periods_x * 2.0 * np.pi, 
               0.1:.1+2.0 * np.pi * periods_y : 1 / sz_y * periods_y * 2.0 * np.pi]

z = np.sin(x)*np.sin(y)
# plt.figure()
# plt.imshow(z)
# datacursor(display='single')

mask = z.copy()
mask[mask <= -.5] = False
mask[mask > 5] = True
mask = np.array(mask, dtype='bool')

#adjust z to PhaseLowerLimit=-.5, PhaseUpperLimit=6.29
z += 1
z /= np.max(np.abs(z))
z *= 4.9
z -= -.5

plt.imshow(z * mask)

signal = 1.0 * np.exp(1j * z)

#simple phase unwrapping
phase = angle(signal)
phase[phase < -.5] += 2.0 * np.pi

plt.imshow(mask * (phase))

np.savez_compressed('/home/tim/repos/py_solvers/application/data/velocity_imaging/2dsine_fully_sampled_data.npz',signal)

np.savez_compressed('/home/tim/repos/py_solvers/application/data/velocity_imaging/2dsine_spatial_mask.npz',mask)

