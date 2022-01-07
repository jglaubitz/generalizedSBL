%% ADMM_2d
%
% Description: 
%  Function that uses ADMM to reconstruct a two-dimensional image based on
%  l1-regulairzation 
% 
% INPUT: 
%  F_1d :           1d forward operator
%  Y :              matrix of indirect measurements 
%  D :              1d regularization matrix 
%  lambda :         regularization parameter 
%  rho, alpha :     ADMM parameters 
%  QUIET :          parameter to surpress or allow display of history 
%
% OUTPUT: 
%  X :              reconstruction 
%  history :        structure containing residual norms and the tolerances at each iteration
%
% Author: Jan Glaubitz 
% Date: Jan 07, 2022
%

function [X, history] = ADMM_2d( F_1d, Y, D, lambda, rho, alpha, QUIET) 

    t_start = tic; % measure time 

    %% Global constants and defaults
    MAX_ITER = 1000;
    ABSTOL   = 1e-4;
    RELTOL   = 1e-2;
    GRAD_DESC_STEPS = 5; 
    
    %% Data preprocessing
    
    % sizes 
    n = size(F_1d,2); % number of pixels in each direction  
    k = size(D,1); % number of outputs of the 1d regularization operator 
    
    % function handles for the matrices and gradient descent 
    fun_FTF = @(X) vec( F_1d'*( ( F_1d*X*(F_1d') ) )*F_1d ); % function that gives the value of F^TAFx 
    fun_RTR = @(X,rho) rho*( vec( D'*( D*X ) ) + vec( ( X*(D') )*D ) ); % function that gives the value of R^TBRx
    fun_G = @(X,rho) fun_FTF(X) + fun_RTR(X,rho); % function that gives value of Gx
    fun_b = @(Y,rho,V1,V2) vec( F_1d'*Y*F_1d ) + rho*vec( V1*D + D'*V2 ); % function that gives the value of b = F^TAy 
    
    %% ADMM solver

    % initialize the (slack) variables
    X = zeros(n,n); X_OLD = zeros(n,n); 
    Z1 = zeros(n,k); Z2 = zeros(k,n); 
    U1 = zeros(n,k); U2 = zeros(k,n); 
    
    % report on process 
    if ~QUIET
        fprintf('%3s\t%10s\t%10s\t%10s\t%10s\n', ... 
            'iter', 'abs error', 'abs tol', 'rel error', 'rel tol');
    end

    % iterate until concergence criterion is satisfied
    for counter = 1:MAX_ITER

        % x-update 
        r = fun_b(Y,rho,Z1-U1,Z2-U2) - fun_G(X,rho); 
        for l=1:GRAD_DESC_STEPS 
            % prepare 
            R = reshape(r,n,n); 
            Gr = fun_G(R,rho); 
            % update rules 
            gamma = norm(r)^2/( dot(r,Gr) ); % step size  
            X = X + gamma*R; % update 
            r = r - gamma*Gr;          
        end

        % z-update with relaxation
        aux1 = alpha*vec(X*(D')) + (1-alpha)*vec(Z1);
        z1 = shrinkage(aux1 + vec(U1), lambda/rho);
        Z1 = reshape(z1,n,k); 
        aux2 = alpha*vec(D*X) + (1-alpha)*vec(Z2);
        z2 = shrinkage(aux2 + vec(U2), lambda/rho);
        Z2 = reshape(z2,k,n); 
        
        % y-update
        U1 = U1 + reshape(aux1,n,k) - Z1; 
        U2 = U2 + reshape(aux2,k,n) - Z2; 

        % store certain values in history structure 
        history.abs_error(counter) = norm( X-X_OLD )^2; % absolute error 
        history.rel_error(counter) = ( norm( X-X_OLD )/norm(X) )^2; % relative error        
        X_OLD = X; % store value of mu 
        
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

function y = shrinkage(a, kappa)
    y = max(0, a-kappa) - max(0, -a-kappa);
end