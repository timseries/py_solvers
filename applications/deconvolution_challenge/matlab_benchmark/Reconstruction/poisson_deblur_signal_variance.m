function [x0,fun_val,QS]=poisson_deblur_signal_variance(y,h,varargin)
import mat_operators.*;
import mat_utils.*;
import mat_utils.results.*;


[x_init,iter,verbose,showfig,tol,img,imgb,K,window,L,ismetric,W,parameters]=...
  process_options(varargin,'x_init',[],'iter',100,'verbose',true,...
  'showfig',false,'tol',1e-5,'img',[],'imgb',[],'K',[0.01 0.03],...
  'window', fspecial('gaussian', 11, 1.5),'L',[],'ismetric',false,'W',[],'parameters',[]);


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

stcObserve = parameters.getSection('Observe1')
b = stcObserve.Background;

if b < 0
  error('RLdeblur3D: Background must have a non-negative value');
else
    disp(num2str(b));
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
epsilon = stcSolver.epsilonStart;
epsilonMin = stcSolver.epsilonStop;
nu = stcSolver.nuStart;
nuMin = stcSolver.nuStop;
decay = stcSolver.decay;
levs=W.get('nlevels');

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

  w_n = W * x;
  N=size(Adjoint(h,y));
  szyb=size(yb);
  yb_padded = zeros(N);
  v=colonvec((N-szyb)/2+1,szyb+(N-szyb)/2);
  yb_padded(v{:})=yb;
  w_y_n=W*yb_padded;
  Yscale=w_y_n.caryScaling;
  S=w_n.copy;
%  alpha=10.0*stcSolver.alpha;
  alpha=0.25*ones(size(stcSolver.alpha));
  alpha(1)=1;
  ary_variance=cell(levs,1);
  for lev=1:levs
      ary_variance_temp=Yscale{lev}./((2*sqrt(2))^(lev));
      ary_variance{lev}=1/8.0*(ary_variance_temp(1:2:end,1:2:end,1:2:end) + ...
                               ary_variance_temp(1:2:end,1:2:end,2:2:end) + ...
                               ary_variance_temp(1:2:end,2:2:end,1:2:end) + ...
                               ary_variance_temp(1:2:end,2:2:end,2:2:end) + ...
                               ary_variance_temp(2:2:end,1:2:end,1:2:end) + ...
                               ary_variance_temp(2:2:end,1:2:end,2:2:end) + ...
                               ary_variance_temp(2:2:end,2:2:end,1:2:end) + ...
                               ary_variance_temp(2:2:end,2:2:end,2:2:end));
  end    
for i=1:iter
%RL step
%  div=y./(Bop(x)+b);
%  div(isnan(div))=0;
%  x=(AdjBop(div).*x)./gamma;



%x is current solution, xnew is update
  for j=1:w_n.J
      lev=max(1,floor((j-2)/28)+1);
%      if j==1
%          var_map=w_y_n{1};
%      else    
%          var_map=ary_variance{lev};
%      end    
      S{j}=1/(0.5.*(abs(w_n{j}).^2) + epsilon^2);
  end
  resid = yb - Direct(h,x);
  
  wr=W * (Adjoint(h,resid));
  for j=1:w_n.J
      lev=max(1,floor((j-2)/28)+1);
      wj=w_n{j};
      wrj=wr{j};
      if j==1
          var_map=w_y_n{1};
      else    
          var_map=ary_variance{lev};
      end    
%      var_map=0;
      wj=((alpha(j)+(nu^2+var_map).*S{j}).^-1).* ... %threshold
         (alpha(j)*(wj)+wrj); %Landweber
      w_n{j}=wj;
  end
  xnew=W'*w_n;
%  xnew(xnew<.15*max(xnew(:)))=0;
  xnew(xnew<b)=0;
  xnewtemp = zeros(N);
  xnewtemp(v{:})=xnew(v{:});
  xnew=xnewtemp;

  w_n=W*xnew;
  epsilon=max(epsilonMin,epsilon*decay)
  nu=max(nuMin,nu*decay)
  
%some hard thresholding
fun_val(i)=cost(y,Bop,xnew,b);
     
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
    fh=sfigure(1);
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
	x0 = Crop(x, y);
end

function Q=cost(y,Bop,f,b)

Hf=Bop(f)+b;
Hf(Hf<0)=0;
%data_term=Hf(:)-y(:).*log(Hf(:));%0*log(0)=0 by convention. Matlab returns 
data_term=norm(Hf(:)+b-y(:),2)^2;%0*log(0)=0 by convention. Matlab returns 
%NAN so we have to fix this. 
data_term(isnan(data_term))=0; 
data_term(isinf(data_term))=0;
%Q=sum(data_term);
Q=data_term;

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