%% ADMM_1d
%
% Description: 
%  Function that uses ADMM to reconstruct a one-dimensional signal based on
%  l1-regulairzation 
% 
% INPUT: 
%  n :      number of (equidistant) grid points
%  gamma : 	blurring parameter  
%
% OUTPUT: 
%  F :      forward operator (real-valued nxn matrix)
%
% Author: Jan Glaubitz 
% Date: Jan 07, 2022
%

function [x, history] = ADMM_1d(F, y, R, lambda, rho, alpha, QUIET) 

    t_start = tic; % measure time 

    %% Global constants and defaults
    MAX_ITER = 1000;
    ABSTOL   = 1e-8;
    RELTOL   = 1e-4;

    %% Data preprocessing
    n = size(F,2);
    k = size(R,1);

    %% ADMM solver

    % initialize the (slack) variables
    x = zeros(n,1); x_OLD = zeros(n,1); 
    z = zeros(k,1);
    u = zeros(k,1);

    % prepare some matrices 
    I = speye(n);
    RtR = R'*R;
    FtF = F'*F;
    
    % report on process 
    if ~QUIET
        fprintf('%3s\t%10s\t%10s\t%10s\t%10s\n', ... 
            'iter', 'abs error', 'abs tol', 'rel error', 'rel tol');
    end

    % iterate until concergence criterion is satisfied
    for counter = 1:MAX_ITER

        % x-update
        x = (FtF + rho*RtR) \ (F.'*y + rho*R'*(z-u));

        % z-update with relaxation
        zold = z;
        Fx_hat = alpha*R*x +(1-alpha)*zold;
        z = shrinkage(Fx_hat + u, lambda/rho);

        % y-update
        u = u + Fx_hat - z;

        % store certain values in history structure 
        history.abs_error(counter) = norm( x-x_OLD )^2; % absolute error 
        history.rel_error(counter) = ( norm( x-x_OLD )/norm(x) )^2; % relative error        
        x_OLD = x; % store value of mu 
        
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

    if ~QUIET
        toc(t_start);
    end

end

function obj = objective(A, b, lambda, x, z)
    obj = .5*norm(A*x - b)^2 + lambda*norm(z,1);
end

function y = shrinkage(a, kappa)
    y = max(0, a-kappa) - max(0, -a-kappa);
end