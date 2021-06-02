%GPLITE_DEMO Demo script with example usage for the GPLITE toolbox.

clear all
rand('twister', 12345)

% Create example data in 1D
N = 20;
D = 2;
X = unifrnd(-3, 3, D, N)';
s2 = zeros(size(X)); 
y = sin(sum(X, 2)) + norminv(rand(N, 1), 0, 0.1);

hyp0 = [];          % Starting hyperparameter vector for optimization
Ns = 10;             % Number of hyperparameter samples
covfun = [3 3];     % GP covariance function
meanfun = 1;        % GP mean function (4 = negative quadratic)
noisefun = [1 0 0]; % Constant plus user-provided noise
hprior = [];        % Prior over hyperparameters
options = [];       % Additional options

% Set prior over noise hyperparameters
gp = gplite_post([],X,y,covfun,meanfun,noisefun,s2,[]);
hprior = gplite_hypprior(gp);

hprior.mu(gp.Ncov+1) = log(1e-3);
hprior.sigma(gp.Ncov+1) = 1;

if gp.Nnoise > 1
    hprior.LB(gp.Ncov+2) = log(5);
    hprior.mu(gp.Ncov+2) = log(10);
    hprior.sigma(gp.Ncov+2) = 0.01;

    hprior.mu(gp.Ncov+3) = log(0.3);
    hprior.sigma(gp.Ncov+3) = 0.01;
    hprior.df(gp.Ncov+3) = Inf;
end

% Train GP on data
[gp,hyp,output] = gplite_train(hyp0,Ns,X,y,covfun,meanfun,noisefun,s2,hprior,options);

hyp            % Hyperparameter sample
 
[xx, yy] = meshgrid(linspace(-5, 5, 20), linspace(-5, 5, 20));
xstar = [reshape(xx.', [], 1), reshape(yy.', [], 1)];
% Compute GP posterior predictive mean and variance at test points
[ymu,ys2,fmu,fs2] = gplite_pred(gp,xstar);

% Plot data and GP prediction
close all;
figure(1); hold on;
gplite_plot(gp);