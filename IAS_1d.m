%% IAS_1d
%
% Description: 
%  Function that solves the MAP estimate of a sparse signal by the IAS algorithm. 
%  We assume a one-dimensional sparse signal and iid noise
%
% INPUT: 
%  F :          forward operator 
%  y :          (indirect) measurements 
%  variance :   noise variance 
%  c, d :       hyper-hyper-parameters 
%
% OUTPUT: 
%  x :         	MAP estimate of the solution 
%  beta :       MAP estimate of the covariance of the prior 
%  history :    structure containing residual norms and the tolerances at each iteration
%
% Author: Jan Glaubitz 
% Date: Jan 07, 2022
% 

function [x, beta, history] = IAS_1d( F, y, variance, c, d, QUIET )

    t_start = tic; % measure time 

    %% Global constants and defaults  
    MIN_ITER = 10; 
    MAX_ITER = 1000; 
    ABSTOL   = 1e-8;
    RELTOL   = 1e-4;
    
    %% Data preprocessing 
    m = size(F,1); % number of (indirect) measurements 
    n = size(F,2); % number of pixels 
    FtF = sparse(F'*F); % product corresponding to the forward operator 
    Fty = F'*y; % forward operator applied to the indirect data 
    
    %% Initial values for the inverse variances and the mean 
    alpha = 1/variance;  
    beta = ones(n,1); 
    x_OLD = zeros(n,1); % mean 
    
    if ~QUIET
        fprintf('%3s\t%10s\t%10s\t%10s\t%10s\n', ... 
            'iter', 'abs error', 'abs tol', 'rel error', 'rel tol');
    end
    
    %% Iterate between the update steps until convergence of max number of iterations 
    for counter = 1:MAX_ITER
        
        % 1) Fix beta and update x 
        D = sparse(diag(1./beta)); 
        C_inv = sparse(alpha*FtF + D); % update covariance matrix  
        x = C_inv\(alpha*Fty); % Least squares solution 
        
        % 2) Fix x and update beta 
        eta = c - 3/2; 
        beta = d*( eta/2 + sqrt( eta^2/4 + x.^2/(2*d) ) );  
        
        % store certain values in history structure 
        history.abs_error(counter) = norm( x-x_OLD )^2; % absolute error 
        history.rel_error(counter) = ( norm( x-x_OLD )/norm(x_OLD) )^2; % relative error        
        x_OLD = x; % store value of mu 
        
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