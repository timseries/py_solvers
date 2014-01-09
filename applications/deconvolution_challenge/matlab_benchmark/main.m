%should be in the top level of these directories in MATLAB current directory (matlab_benchmark)
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Operators/'])
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Reconstruction/'])
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Metrics/'])
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Metrics/FourierMetrics'])
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Misc/'])

str_dir = '/media/sf_Google_Drive/PhD/Projects/DeconvolutionChallenge/Data/P0/';
%str_measurements = 'Measurements/';
%str_psfs = 'PSFs/';
str_measurements = '';
str_psfs = '';
str_case = 'phantom';
str_case_padded = 'phantom_padded';
ary_ground_truth = imreadstack([str_dir str_measurements str_case '.tif']);
ary_ground_truth_padded = imreadstack([str_dir str_measurements str_case_padded '.tif']);
%ary_psf = imreadstack([str_dir str_psfs str_case '_psf.tif']);
ary_psf = imreadstack([str_dir str_psfs 'psf.tif']);
%0
mp = 289.8;
b = 11.2;
stdev = 5.6;
seed = 1;
%observe

[y,f,fb]=ForwardModel3D(ary_ground_truth_padded,ary_psf,mp,b,stdev,seed);
%reverse quantize the measurements y
%y = double(ary_ground_truth);
y = double(y);

[x0,fun_val,QS]=RLdeblur3D(y,ary_psf,'img',f,'imgb',fb,'iter',100,'verbose',true,'showfig',false,'ismetric',true);
%[x0,fun_val,QS]=RLdeblur3D(ary_ground_truth,ary_psf,'iter',30,'verbose',true,'showfig',true);
implay(uint8(x0));
