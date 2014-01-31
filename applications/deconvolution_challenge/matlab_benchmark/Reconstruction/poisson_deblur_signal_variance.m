function [x0,fun_val,QS]=poisson_deblur_signal_variance(y,h,varargin)
import mat_operators.*;
import mat_utils.*;

%Image deblurring with Richardson-Lucy algorithm.

% ========================== INPUT PARAMETERS (required) ==================
% Parameters    Values  description
% =========================================================================
% y             Noisy blured image.
% h             Blur kernel (PSF).
% ======================== OPTIONAL INPUT PARAMETERS ======================
% Parameters    Values description
% =========================================================================
% img           Original Image. (For the compution of the Quality metrics)
% imgb          Blurred Image (For the compution of the Normalized mean 
%               integrated squared error). It is the third output argument
%               of the ForwardModel3D.m
% x_init        Initial guess for x. (Default: [])
% iter          Number of iterations (Default: 100)
% tol           Stopping threshold: Relative normed difference between two
%               successive iterations (Default:1e-5)
% verbose       If verbose is set on then info for each iteration is
%               printed on screen. (Default: false)
% showfig       If showfig is set to true, the maximum intensity projection 
%               of the result of the deconvolution in each iteration is 
%               shown on screen. (Default: false)
% b             Constant value for the background (Default: 0)
% K             constants in the SSIM index formula (see ssim_index.m).
%               default value: K = [0.01 0.03]
% window        local window for statistics (see ssim_index.m). default 
%               window is Gaussian given by window = fspecial('gaussian', 11, 1.5);
% L             dynamic range of the images. default: max(X2(:))-min(X2(:))
% ismetric      If ismetric is set to true then the quality metrics are
%               computed for every iteration. (Default: false) (Note: If
%               this option is set to true the script can be very slow)
% =========================================================================

%Author: stamatis.lefkimmiatis@epfl.ch (Biomedical Imaging Group)

[x_init,iter,verbose,showfig,tol,img,imgb,b,K,window,L,ismetric,nu,epsilon,decay,W]=...
  process_options(varargin,'x_init',[],'iter',100,'verbose',true,...
  'showfig',false,'tol',1e-5,'img',[],'imgb',[],'b',0,'K',[0.01 0.03],...
  'window', fspecial('gaussian', 11, 1.5),'L',[],'ismetric',false,'nu',[],'epsilon',[],'decay',[],'W',[]);


if nargout > 2 && ismetric
  nbins=64;
  Z=zeros(iter,1);
  img_hat=fftn(img);
  metric=FourierMetricsConstructor(size(y),nbins);
  QS=struct('ISNR',Z,'RSNR',Z,'nmise_metric',Z,'ssim_metric',[Z Z],...
    'FSC_metric',repmat(Z,1,nbins),'ER_metric',repmat(Z,1,nbins));  
else
  QS=[];
end


if b < 0
  error('RLdeblur3D: Background must have a non-negative value');
end


% Operators
Bop = @(x) Direct(h, x); % Function handle that corresponds to the blurring operator
AdjBop = @(y) Adjoint(h, y); % Function handle that corresponds to the adjoint of the blurring operator

%x0 Initialization
if isempty(x_init)
    x_init=Adjoint(h, y);
    
end
size(x_init)
x=x_init;

%stuff for nonlinear solver, won't change from iteration to the next
stcSolver = parameters.getSection('Solver1')
b = stcSolver.b;
epsilon = stcSolver.epsilonStart;
epsilonMin = stcSolver.epsilonStop;
nu = stcSolver.nuStart;
nuMin = stcSolver.nuStop;
decay = stcSolver.decay;

pars.h = h;
pars.W = W;
pars.epsilon = epsilon;
pars.w_n = W * x;
pars.nvar = pars.w_n.getNumCoeffs()
pars.fgname = 'fungrad';
options.maxit=10;
options.version='C';
options.prtlevel=2;

% Normalization constant, probably won't need this
gamma = AdjBop(ones(size(y)));

fun_val=zeros(iter,1);

count=0;
if verbose
  fprintf('\t\t****************************************\n');
  fprintf('\t\t** Deconvolution with Richardson-Lucy **\n');
  fprintf('\t\t****************************************\n');
  fprintf('#iter       fun-val      relative-dif     ISNR\t  RSNR\t   NMISE\t  SSIM_mean\t   SSIM_min\n')
  fprintf('==========================================================================================\n');
end

yb=y-b;

for i=1:iter
%x is current solution, xnew is update
  w_n = W * x;
  
  xnew = W' * w_n;
  wvec_n = pars.w_n.getSubbandsArray()
%fun_val(i)=cost(y,Bop,xnew,b);
     
  re=norm(xnew(:)-x(:))/norm(x(:));%relative error
  if (re < tol)
    count=count+1;
  else
    count=0;
  end
  
  
  if verbose 
    if ~isempty(img) && ismetric
	  x0 = Crop(xnew, y);
      if isempty(L)
        L=max(img(:))-min(img(:));
      end

	  xreg = AffineRegression(x0, img);
      QS.ISNR(i)=20*log10(norm(y(:)-img(:))/norm(x0(:)-img(:)));
      QS.RSNR(i)=20*log10(norm(y(:)-img(:))/norm(xreg(:)-img(:)));
      QS.RSNR(i)=0;
      if isempty(imgb)
        QS.nmise_metric(i)=nan;
      else
        QS.nmise_metric(i)=nmisec(x0,img,imgb);
      end
      [ssim_mean,ssim_min]=ssim3D(x0,img,K,window,L);
      QS.ssim_metric(i,:)=[ssim_mean, ssim_min];
      x0_hat=fftn(x0);
      QS.FSC_metric(i,:)=FourierShellCorrelation(metric,x0_hat,img_hat);
      QS.ER_metric(i,:)=RelativeEnergyRegain(metric,x0_hat,img_hat);
      
      % printing the information of the current iteration
      fprintf('%3d \t %3.5f \t %3.5f \t %3.5f \t %3.5f \t%3.5f \t%3.5f \t%3.5f\n',i,...
      fun_val(i),re,QS.ISNR(i),QS.RSNR(i),QS.nmise_metric(i),QS.ssim_metric(i,1),QS.ssim_metric(i,2));
    else
      fprintf('%3d \t %3.5f \t %3.5f\n',i,fun_val(i),re);
      %fprintf('%3d \t %3.5f\t%1.5f\n',i,fun_val(i),re);
    end
  end
  x=xnew; 
    
  if showfig
    fh=figure(1);
    figure(fh);
    msg=['iteration: ' num2str(i) ' ,Objective Value: ' num2str(fun_val(i))];
    set(fh,'name',msg);imshow(max(x,[],3),[]);
  end
  
  if count >=5
    fun_val(i+1:end)=[];
    break;
  end
  
end

if ~exist('x0', 'var');
	x0 = Crop(y, x);
end

function Q=cost(y,Bop,f,b)

Hf=Bop(f)+b;
Hf(Hf<0)=0;
data_term=Hf(:)-y(:).*log(Hf(:));%0*log(0)=0 by convention. Matlab returns 
%NAN so we have to fix this. 
data_term(isnan(data_term))=0; 
data_term(isinf(data_term))=0;
Q=sum(data_term);

function res=u(w_n,b,h,W)
    f = f(w_n,b,h,W);
    res = f - 1 / (2 * f);
    
function res=u_prime(w_n,b,h,W)    
    f = f(w_n,b,h,W);
    res = 2/f + 1 / f^3;
    
function res=f(w_n,b,h,W)
    b1 = b + 3 / 8;
    res = 2*sqrt(Direct(h,(W'*w_n))+b1);
    
%function [val,grad] = fungrad(wvec_n, pars)
%    pars.w_n.setSubbandsArray(wvec_n,1:pars.w_n.J,1);
%    u = u(pars.w_n,pars.b,pars.h,pars.W);
%    u_prime = u_prime(pars.w_n,pars.b,pars.h,pars.W);
%%compute S here
%    S=S(wvec_n,pars.epsilon);
%%yield fn val
%    val = (norm(u(:)-pars.q(:),2)^2 + (wvec_n.^2).*S)/2;
%    gradw = pars.W * Adjoint(pars.h, (u_prime.*(u-pars.q)));
%    grad = gradw.getSubbandsArray() + wvec_n.*S;
    
function res = S(wvec_n,epsilon)
    Stemp = zeros(size(wvec_n));
    Stemp(1:2:end) = .5/(wvec_n(1:2:end).^2+wvec_n(2:2:end).^2+epsilon^2);
    Stemp(2:2:end) = Stemp(1:2:end);
    res = Stemp;