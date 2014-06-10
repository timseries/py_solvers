%should be in the top level of these directories in MATLAB current directory (matlab_benchmark)
%get the wavelet transform operator
import mat_operators.*;
import mat_utils.*;
import mat_utils.results.*;
psParameters=ParameterStruct('/home/tim/repos/py_solvers/applications/deconvolution_challenge/p1.ini');
W=operatorFactory(psParameters,'Transform1');
stcInput1 = psParameters.getSection('Input1');
stcInput2 = psParameters.getSection('Input2');

addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/nlcg1_0/'])
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Operators/'])
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Reconstruction/'])
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Metrics/'])
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/' ...
         'Metrics/FourierMetrics'])
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/' ...
         'Misc/'])
addpath(['/home/tim/repos/scratch/dtcwt'])
addpath(['/home/tim/repos/py_operators/dtcwt/matlab/qbgn'])

str_eval_result_dir='/home/tim/repos/py_solvers/applications/deconvolution_challenge/eval_results/';
%str_dir = '/media/OS/Documents and Settings/tim/Google Drive/PhD/Projects/DeconvolutionChallenge/Data/P0/';
str_dir = stcInput1.FileDir;

str_case_padded=stcInput1.FileName;
%str_measurements = 'Measurements/';
%str_psfs = 'PSFs/';
%phantoms
str_measurements = '';
str_psfs = '';
str_psf = stcInput2.FileName;
str_case = 'phantom';
%str_case_padded = 'phantom_padded';
ary_ground_truth = imreadstack([str_dir str_measurements str_case '.tif']);
ary_ground_truth_padded = imreadstack([str_dir str_measurements str_case_padded]);
%str_dir = '/media/OS/Documents and Settings/tim/Google Drive/PhD/Projects/DeconvolutionChallenge/Data/P0/';
%challenge data
%str_measurements = '';
%str_psfs = '3_';
%str_case_padded = '3';
%y = imreadstack([str_dir str_measurements str_case_padded '.tif']);

%ary_psf = imreadstack([str_dir str_psfs str_case '_psf.tif']);
ary_psf = imreadstack([str_dir str_psfs str_psf]);
%0
% mp = 239.6;
% b = 15.8;
% stdev = 9.7;
% seed = 1;

nu = 9;
epsilon = 20;
decay=.9;
%1
mp = 342.4;
b = 7.8;
stdev = 3.1;
seed = 1;

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

%%%%%%PARAMS FOR EVALUATION%%%%%%%%




%observe
[y,f,fb,r]=ForwardModel3D(ary_ground_truth_padded,ary_psf,mp,b,stdev,seed);
%reverse quantize the measurements y
%y = double(ary_ground_truth);


y = double(y);

%figure;hist(ary_ground_truth(:),100);title('x');figure;hist(r(:),100);title('mp/max(Ax)*Ax');figure;hist(f(:),100);title('mp/max(Ax)*x');figure;hist(fb(:),100);title('mp/max(Ax)*Ax+b');

W.lgcAdjoint=0;
%[x0,fun_val,QS]=sparse_poisson_deblur(y,ary_psf,'img',f,'imgb',fb,'iter',100,'verbose',true,'showfig',false,'ismetric',true,'nu',nu,'epsilon',epsilon,'decay',decay,'W',W,'b',b); 
sfigure;[x0,fun_val,QS]=RLdeblur3D(y,ary_psf,'img',f,'imgb',fb,'iter',70,'verbose',true,'showfig',true,'ismetric',true,'nu',nu,'epsilon',epsilon,'decay',decay,'W',W,'b',b); 
% [x0,fun_val,QS]=poisson_deblur_signal_variance(y,ary_psf,'img',f,'imgb',fb,'iter',100,'verbose',true,'showfig',true,'ismetric',true,'W',W,'parameters',psParameters); 


%for eval
%y=ary_ground_truth_padded;
%[x0,fun_val,QS]=poisson_deblur_signal_variance(y,ary_psf,'iter',150,'verbose',true,'showfig',true,'ismetric',true,'W',W,'parameters',psParameters); 
%[x0,fun_val,QS]=RLdeblur3D(ary_ground_truth,ary_psf,'iter',30,'verbose',true,'showfig',true);
implay(uint8(x0));
