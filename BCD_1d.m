%% BCD_1d
%
% Description: 
%  Function that performs SBL based on the Bayesian coordinate descent approach. 
%  We assume a one-dimensional signal and iid noise
%
% INPUT: 
%  F :      forward operator 
%  y :    	(indirect) measurements 
%  R :     	regularization matrix
%  c, d :  	hyper-hyper-parameters 
%
% OUTPUT: 
%  mu :         mean of the posterior 
%  C_inv :     	inverse covariance matrix of the posterior  
%  alpha :      inverse covariance of the noise 
%  beta :       inverse covariance of the prior 
%  history :    structure containing residual norms and the tolerances at each iteration
%
% Author: Jan Glaubitz 
% Date: Jan 07, 2022
% 

function [mu, C_inv, alpha, beta, history] = BCD_1d( F, y, R, c, d, QUIET )

    t_start = tic; % measure time 

    %% Global constants and defaults  
    MIN_ITER = 10; 
    MAX_ITER = 1000; 
    ABSTOL   = 1e-8;
    RELTOL   = 1e-4;
    
    %% Data preprocessing 
    m = size(F,1); % number of (indirect) measurements 
    n = size(F,2); % number of pixels 
    k = size(R,1); % number of outputs of the regularization operator 
    FtF = sparse(F'*F); % product corresponding to the forward operator 
    Fty = F'*y; % forward operator applied to the indirect data 
    
    %% Initial values for the inverse variances and the mean 
    alpha = 1; 
    beta = ones(k,1); 
    mu_OLD = zeros(n,1); % mean 
    
    if ~QUIET
        fprintf('%3s\t%10s\t%10s\t%10s\t%10s\n', ... 
            'iter', 'abs error', 'abs tol', 'rel error', 'rel tol');
    end
    
    %% Iterate between the update steps until convergence of max number of iterations 
    for counter = 1:MAX_ITER
        
        % 1) Fix alpha, beta, and update x 
        B = sparse(diag(beta)); % inverse noise variance 
        C_inv = sparse(alpha*FtF + R'*B*R); % update covariance matrix  
        mu = C_inv\(alpha*Fty); % update the mean 
        
        % 2) Fix x, B and update alpha 
        alpha = ( m + 2*c)./( norm(F*mu-y)^2 + 2*d );  
        
        % 3) Fix x, alpha and update B 
        beta = (1+2*c)./( (R*mu).^2 + 2*d ); 
        
        % store certain values in history structure 
        history.abs_error(counter) = norm( mu-mu_OLD )^2; % absolute error 
        history.rel_error(counter) = ( norm( mu-mu_OLD )/norm(mu_OLD) )^2; % relative error        
        mu_OLD = mu; % store value of mu 
        
        % display these values if desired 
        if ~QUIET
            fprintf('%3d\t%0.2e\t%0.2e\t%0.2e\t%0.2e\n', ... 
                counter, history.abs_error(counter), ABSTOL, ... 
                history.rel_error(counter), RELTOL);
        end
        
        % check for convergence 
        if ( history.abs_error(counter) < ABSTOL && ...
                history.rel_error(counter) < RELTOL && ... 
                counter > MIN_ITER )
             break;
        end
        
    end

    % output the time it took to perform all operations 
    if ~QUIET
        toc(t_start);
    end
    
end