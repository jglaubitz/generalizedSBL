%% BCD_1d_fusion
%
% Description: 
%  Function that performs SBL based on the Bayesian coordinate descent approach. 
%  We assume a one-dimensional signal and two different data sources. 
% 
% INPUT: 
%  F1, F2 :             forward operators
%  y1, y2 :             (indirect) measurements 
%  R :                  regularization matrix
%  c, d :               hyper-hyper-parameters 
%
% OUTPUT: 
%  mu :         mean of the posterior 
%  C_inv :          inverse covariance matrix of the posterior 
%  alpha :      inverse covariance of the noise 
%  beta :       inverse covariance of the prior 
%  history :    structure containing residual norms and the tolerances at each iteration
%
% Author: Jan Glaubitz 
% Date: Jan 07, 2022
% 

function [mu, C_inv, alpha, beta, history] = BCD_1d_fusion( F1, F2, y1, y2, R, c, d, QUIET )

    t_start = tic; % measure time 

    %% Global constants and defaults 
    MIN_ITER = 10; 
    MAX_ITER = 1000; 
    ABSTOL   = 1e-8;
    RELTOL   = 1e-4;
    
    %% Data preprocessing 
    m1 = size(F1,1); % number of (indirect) measurements 
    m2 = size(F2,1); % number of (indirect) measurements 
    n = size(F1,2); % number of pixels 
    k = size(R,1); % number of outputs of the regularization operator 
    F1tF1 = sparse(F1'*F1); % product corresponding to the forward operator 
    F2tF2 = sparse(F2'*F2); % product corresponding to the forward operator 
    F1ty = F1'*y1; % forward operator applied to the indirect data 
    F2ty = F2'*y2; % forward operator applied to the indirect data 
    
    %% Initial values for the inverse variances and the mean 
    alpha1 = 1; alpha2 = 1; 
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
        C_inv = sparse(alpha1*F1tF1 + alpha2*F2tF2 + R'*B*R); % update covariance matrix  
        mu = C_inv\(alpha1*F1ty+alpha2*F2ty); % update the mean
        
        % 2) Fix x, B and update alpha 
        alpha1 = ( m1 + 2*c)./( norm(F1*mu-y1)^2 + 2*d ); 
        alpha2 = ( m2 + 2*c)./( norm(F2*mu-y2)^2 + 2*d );
        alpha = [alpha1,alpha2]; 
        
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
           history.rel_error(counter) < RELTOL )
             break;
        end
        
    end

    % output the time it took to perform all operations 
    if ~QUIET
        toc(t_start);
    end
    
end