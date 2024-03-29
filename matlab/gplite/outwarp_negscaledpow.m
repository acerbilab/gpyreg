function [ywarp,dwarp_dt,dwarp_dtheta,d2warp_dthetadt] = outwarp_negscaledpow(hyp,y,invflag)
%GPLITE_NOISEFUN Noise function for lite Gaussian Process regression.
%   SN2 = GPLITE_NOISEFUN(HYP,X,NOISEFUN) computes the GP noise function
%   NOISEFUN, that is the variance of observation noise evaluated at test
%   points X. HYP is a single column vector of noise function
%   hyperparameters. NOISEFUN is a numeric array whose elements specify
%   features of the noise function, as follows:
%
%   See also GPLITE_COVFUN, GPLITE_MEANFUN.

if nargin < 2; y = []; end
if nargin < 3 || isempty(invflag); invflag = false; else; invflag = true; end

if invflag && nargout > 1
    error('outwarp_fun:InverseOnly', ...
        ['When calling for the inverse output warping function, only one function output is expected.']);
end

%--------------------------------------------------------------------------
% CUSTOM: Number of hyperparameters
Noutwarp = 3;       % # hyperparameters of the output warping function
%--------------------------------------------------------------------------

N = size(y,1);      % Number of training points

% Return number of output warping function hyperparameters and additional info
if ischar(hyp)
    ywarp = Noutwarp;
    if nargout > 1

        % Initialize bounds for all hyperparameters
        outwarp_info.LB = -Inf(1,Noutwarp);
        outwarp_info.UB = Inf(1,Noutwarp);
        outwarp_info.PLB = -Inf(1,Noutwarp);
        outwarp_info.PUB = Inf(1,Noutwarp);
        outwarp_info.x0 = NaN(1,Noutwarp);

        %------------------------------------------------------------------
        % CUSTOM: Initialize hyperparameter bounds and other details

        % Threshold parameter
        outwarp_info.LB(1) = min(y);
        outwarp_info.UB(1) = max(y);
        outwarp_info.PLB(1) = min(y);
        outwarp_info.PUB(1) = max(y);
        outwarp_info.x0(1) = NaN;

        % Scaling parameter a (log space)
        outwarp_info.LB(2) = -Inf;
        outwarp_info.UB(2) = Inf;
        outwarp_info.PLB(2) = -2;
        outwarp_info.PUB(2) = 2;
        outwarp_info.x0(2) = 0;

        % Power exponent k (log space)
        outwarp_info.LB(3) = -Inf;
        outwarp_info.UB(3) = Inf;
        outwarp_info.PLB(3) = -3;
        outwarp_info.PUB(3) = 3;
        outwarp_info.x0(3) = 0;

        %------------------------------------------------------------------

        % Assign handle of current output warping function
        outwarp_info.outwarpfun = str2func(mfilename);

        % Plausible starting point
        idx_nan = isnan(outwarp_info.x0);
        outwarp_info.x0(idx_nan) = 0.5*(outwarp_info.PLB(idx_nan) + outwarp_info.PUB(idx_nan));

        dwarp_dt = outwarp_info;

    end

    return;
end

[Nhyp,Ns] = size(hyp);      % Hyperparameters and samples

if Nhyp ~= Noutwarp
    error('outwarp_fun:WrongLikHyp', ...
        ['Expected ' num2str(Noutwarp) ' output warping function hyperparameters, ' num2str(Nhyp) ' passed instead.']);
end
if Ns > 1
    error('outwarp_fun:nosampling', ...
        'Output warping function output is available only for one-sample hyperparameter inputs.');
end

%--------------------------------------------------------------------------
% CUSTOM: Compute output warping function and gradients

% Read hyperparameters
y0 = hyp(1);
a = exp(hyp(2));
k = exp(hyp(3));

% Compute output warping or inverse warping
ywarp = y;
idx = y < y0;
if invflag      % Inverse output warping
    ywarp(idx) = y0 - ((y0 - y(idx)).^(1/k))/a;
else            % Direct output warping
    adelta = a*(y0 - y(idx));
    adeltak = adelta.^k;
    ywarp(idx) = y0 - adeltak;
end

if nargout > 1
    % First-order derivative of output warping function in output space
    dwarp_dt = ones(size(y));
    adeltakm1 = adelta.^(k-1);

    dwarp_dt(idx) = a*k*adeltakm1;

    if nargout > 2
        % Gradient of output warping function wrt hyperparameters
        dwarp_dtheta = zeros(N,Noutwarp);

        dwarp_dtheta(idx,1) = 1 - a*k*adeltakm1;            % y0
        dwarp_dtheta(idx,2) = -k*adeltak;                   % log(a)
        dwarp_dtheta(idx,3) = -k*adeltak.*log(adelta);      % log(k)

        if nargout > 3
            % Gradient of derivative of output warping function
            d2warp_dthetadt = zeros(N,Noutwarp);

            d2warp_dthetadt(idx,1) = a^2*k*(k-1)*adelta.^(k-2);                % y0
            d2warp_dthetadt(idx,2) = a*k^2*adeltakm1;                           % log(a)
            d2warp_dthetadt(idx,3) = a*k*adeltakm1 + a*k^2*adeltakm1.*log(adelta);  % log(k)

        end

    end
end

end
