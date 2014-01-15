function [val,grad] = fungrad(wvec_n, pars)
    pars.w_n.setSubbandsArray(wvec_n);
    u = u(pars.w_n,pars.b,pars.h,pars.W);
    u_prime = u_prime(pars.w_n,pars.b,pars.h,pars.W);
%compute S here
    S=S(wvec_n,pars.epsilon);
%yield fn val
    val = (norm(u(:)-pars.q(:),2)^2 + (wvec_n.^2).*S)/2;
    gradw = pars.W * Adjoint(pars.h, (u_prime.*(u-pars.q)));
    grad = gradw.getSubbandsArray() + wvec_n.*S;
    
function res=u(w_n,b,h,W)
    f = f(w_n,b,h,W);
    res = f - 1 / (2 * f);
    
function res=u_prime(w_n,b,h,W)    
    f = f(w_n,b,h,W);
    res = 2/f + 1 / f^3;
    
function res=f(w_n,b,h,W)
    b1 = b + 3 / 8;
    res = 2*sqrt(Direct(h,(W'*w_n))+b1);
    
    
function res = S(wvec_n,epsilon)
    res = .5/(wvec_n.^2+epsilon^2)