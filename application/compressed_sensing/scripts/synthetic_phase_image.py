#script to generate a mask boundary 
import matplotlib.pyplot as plt
import numpy as np
from numpy import abs as nabs, log
from numpy.fft import fftn, fftshift
from mpldatacursor import datacursor
from py_utils.signal_utilities.sig_utils import gaussian
from scipy.signal import convolve, firwin

#2d sinusoid, filtered noise parameters, and phase scaling
periods_x = 3.0
periods_y = 3.0

gauss_szx = 19.0
gauss_szy = 19.0

imszx = 168
imszy = 168
sz_x = imszx+gauss_szx-1
sz_y = imszy+gauss_szy-1

gauss_sigma = 1.5
gauss_filter = gaussian((gauss_szx, gauss_szy),(gauss_sigma,gauss_sigma))

x = np.arange(sz_x)/sz_x*periods_x*2.0*np.pi
y = np.arange(sz_y)/sz_y*periods_y*2.0*np.pi

x,y = np.mgrid[0:2.0 * np.pi * periods_x : 1 / sz_x * periods_x * 2.0 * np.pi, 
               0:2.0 * np.pi * periods_y : 1 / sz_y * periods_y * 2.0 * np.pi]

z = np.sin(x)*np.sin(y)

max_phase = 4.0
min_phase = -1.0
# min_phase = 0.0

plt.imshow(z)
plt.figure()
zfft = fftshift(fftn(z))
plt.imshow(log(abs(zfft)))
datacursor(display='single')
#start with 
# z=0
gauss_random = np.random.rand(int(sz_x),int(sz_y))
gauss_random = gauss_random / np.max(np.abs(gauss_random))
z += gauss_random

#now blur with a Gaussian

zsmooth = convolve(z, gauss_filter,mode='valid')

if np.min(zsmooth) < 0:
    zsmooth += nabs(np.min(zsmooth))
zsmooth /= np.max(np.abs(zsmooth))
zsmooth *= max_phase
zsmooth -= (np.min(zsmooth) - min_phase)

plt.imshow(zsmooth)
colorbar()
# datacursor(display='single')

mask = zsmooth.copy()
mask[mask <= (0.0)] = False
mask[mask > (0.0)] = True
mask = np.array(mask, dtype='bool')
plt.imshow(mask)
plt.imshow(zsmooth*mask)

zsmooth_fft = fftshift(fftn(zsmooth))
plt.imshow(log(nabs(zsmooth_fft)))

# plt.imshow(log(nabs(fftshift(fftn(dict_in['theta'])))))

# plt.figure()
# x_vel_enc = np.load('/home/tim/repos/py_solvers/application/data/velocity_imaging/reducing_times_benchmark_fully_sampled_phase.npz')['arr_0']

# vel_enc_phase_no_mask = angle(dict_in['x'])
# vel_enc_phase_no_mask[vel_enc_phase_no_mask < -.5] += 2*pi
# np.savez_compressed('/home/tim/repos/py_solvers/application/data/velocity_imaging/reducing_times_benchmark_fully_sampled_phase_no_mask.npz',vel_enc_phase_no_mask)
# plt.imshow(log(nabs(fftshift(fftn(x_vel_enc)))))


cx,cy = mgrid[0:imszx,0:imszy]
circle_dist = (cx-imszx/2)**2 + (cy-imszy/2)**2
circle = (circle_dist < (120-imszx/2)**2)
plt.imshow(log(nabs(zsmooth_fft))*circle)
# zsmooth_lpf_fft = zsmooth_fft*circle
zsmooth_lpf_fft = zsmooth_fft*1 #don't need lpf anymore, but if we do uncomment the line above, and comment this on out

zsmooth_lpf = np.real(ifftn(ifftshift(zsmooth_lpf_fft)))
plt.imshow(zsmooth_lpf, cmap='pink')
plt.colorbar()

plt.figure()
plt.imshow(log(abs(fftshift(fftn(zsmooth_lpf)))))
plt.figure()
# plt.imshow(log(abs(fftshift(fftn(zsmooth_lpf*mask)))))
# plt.figure()
plt.imshow(mask * zsmooth_lpf,cmap='pink')
plt.colorbar()
# signal = 1.0 * np.exp(1j * zsmooth_lpf)

#simple phase unwrapping
# phase = angle(signal)
phase = mask * zsmooth_lpf
# phase[phase < min_phase] += 2.0 * np.pi
# phase[phase > max_phase] -= 2.0 * np.pi

signal = (1.0-mask) * np.exp(1j * phase)

plt.imshow(phase,cmap='pink')
plt.colorbar()

line_vel=phase[40,:]
plt.plot(line_vel)
plt.figure()
plt.imshow(log(nabs(fftshift(fftn(phase)))))

plt.imshow(np.abs(signal))
np.savez_compressed('/home/tim/repos/py_solvers/application/data/velocity_imaging/2dsine_fully_sampled_data.npz',signal)

np.savez_compressed('/home/tim/repos/py_solvers/application/data/velocity_imaging/2dsine_spatial_mask.npz',mask)


