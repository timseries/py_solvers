%should be in the top level of these directories in MATLAB current directory (matlab_benchmark)
%get the wavelet transform operator
import mat_operators.*;
import mat_utils.*;
psParameters=ParameterStruct('/home/tim/repos/py_solvers/applications/deconvolution_challenge/p0.ini');
W=operatorFactory(psParameters,'Transform1');

addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/nlcg1_0/'])
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Operators/'])
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Reconstruction/'])
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Metrics/'])
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Metrics/FourierMetrics'])
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Misc/'])

str_dir = '/media/sf_Google_Drive/PhD/Projects/DeconvolutionChallenge/Data/P0/';
%str_measurements = 'Measurements/';
%str_psfs = 'PSFs/';
%phantoms
str_measurements = '';
str_psfs = '';
str_case = 'phantom';
str_case_padded = 'phantom_padded';
ary_ground_truth = imreadstack([str_dir str_measurements str_case '.tif']);
ary_ground_truth_padded = imreadstack([str_dir str_measurements str_case_padded '.tif']);

%challenge data
%str_measurements = '';
%str_psfs = '3_';
%str_case_padded = '3';
%y = imreadstack([str_dir str_measurements str_case_padded '.tif']);

%ary_psf = imreadstack([str_dir str_psfs str_case '_psf.tif']);
ary_psf = imreadstack([str_dir str_psfs 'psf.tif']);
%0
mp = 239.6;
b = 15.8;
stdev = 9.7;
seed = 1;

nu = 9;
epsilon = 20;
decay=.9;
%1
%mp = 342.4;
%b = 7.8;
%stdev = 3.1;
%seed = 1;

%2
%mp = 289.8;
%b = 11.2;
%stdev = 5.6;
%seed = 1;

%3
%mp = 153.8;
%b = 19.6;
%stdev = 12.3;
%seed = 1;


%observe

[y,f,fb]=ForwardModel3D(ary_ground_truth_padded,ary_psf,mp,b,stdev,seed);
%reverse quantize the measurements y
%y = double(ary_ground_truth);
y = double(y);

W.lgcAdjoint=0;
[x0,fun_val,QS]=sparse_poisson_deblur(y,ary_psf,'img',f,'imgb',fb,'iter',100,'verbose',true,'showfig',false,'ismetric',true,'nu',nu,'epsilon',epsilon,'decay',decay,'W',W,'b',b);
%[x0,fun_val,QS]=RLdeblur3D(ary_ground_truth,ary_psf,'iter',30,'verbose',true,'showfig',true);
implay(uint8(x0));
