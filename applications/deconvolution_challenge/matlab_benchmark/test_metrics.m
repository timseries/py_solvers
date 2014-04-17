addpath('~/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Metrics/')
addpath('~/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Metrics/FourierMetrics/')
addpath('~/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/FourierMetrics/')
addpath('~/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Misc/')
str_input_file='~/repos/scratch/deconv_challenge_output/p0/011/p0_x_n0.tif';
x_n0=imreadstack(str_input_file);
str_input_file='~/repos/scratch/deconv_challenge_output/p0/011/p0_x.tif';
x=imreadstack(str_input_file);
str_input_file='~/repos/scratch/deconv_challenge_output/p0/011/p0_fb.tif';
fb=imreadstack(str_input_file);
str_input_file='~/repos/scratch/deconv_challenge_output/p0/011/p0_y.tif';
y=imreadstack(str_input_file);

K = [0.01 0.03];
window=fspecial('gaussian', 11, 1.5);
L=max(x(:))-min(x(:));
nbins=64;

disp('nmisec')
nmisec(x_n0,x,fb)
[ssim_mean,ssim_min] = ssim3D(x_n0, x, K, window, L)
[snr_db, MSE] = psnr(x_n0, x)
metric=FourierMetricsConstructor(size(y),nbins);


fsc= FourierShellCorrelation(metric,fftn(x_n0),fftn(x))
rer = RelativeEnergyRegain(metric, fftn(x_n0),fftn(x))
