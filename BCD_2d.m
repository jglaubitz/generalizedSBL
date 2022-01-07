%% BCD_2d
%
% Description: 
%  Function that performs Bayesian learning based on a Bayesian 
%  coordinate descent approach 
%
% INPUT: 
%  F_1d :               1d forward operator
%  Y :                  matrix of indirect measurements 
%  D :                  1d regularization matrix
%  c, d :               hyper-hyper-parameters 
%
% OUTPUT: 
%  Mu :         mean of the posterior 
%  alpha :      inverse covariance of the noise 
%  B1, B2 :     inverse covariances of the prior 
%  history :    structure containing residual norms and the tolerances at each iteration
%
% Author: Jan Glaubitz 
% Date: Jan 07, 2022
% 

function [Mu, alpha, B1, B2, history] = BI_BayCD_2d_iid( F_1d, Y, D, c, d )

    t_start = tic; % measure time 

    %% Global constants and defaults 
    QUIET    = 0; 
    MIN_ITER = 10; 
    MAX_ITER = 1000; 
    ABSTOL   = 1e-4;
    RELTOL   = 1e-2;
    GRAD_DESC_STEPS = 5; 
    
    %% Data preprocessing 
    m = size(F_1d,1); % number of (indirect) measurements 
    n = size(F_1d,2); % number of pixels in each direction  
    k = size(D,1); % number of outputs of the regularization operator 
    fun_FTAF = @(X,alpha) alpha*vec( F_1d'*( ( F_1d*X*(F_1d') ) )*F_1d ); % function that gives the value of F^TAFx 
    fun_RTBR = @(X,B1,B2) vec( D'*( B1.*(D*X) ) ) + ... 
        vec( ( B2.*(X*(D')) )*D ); % function that gives the value of R^TBRx
    fun_G = @(X,alpha,B1,B2) fun_FTAF(X,alpha) + fun_RTBR(X,B1,B2); % function that gives value of Gx
    fun_b = @(Y,alpha) alpha*vec( F_1d'*Y*F_1d ); % function that gives the value of b = F^TAy 
    
    %% Initial values for the inverse variances and the mean 
    alpha = 1;
    B1 = ones(k,n); B2 = ones(n,k);
    Mu = zeros(n,n); 
    Mu_OLD = Mu; % mean 
    
    if ~QUIET
        fprintf('%3s\t%10s\t%10s\t%10s\t%10s\n', ... 
            'iter', 'abs error', 'abs tol', 'rel error', 'rel tol');
    end
    
    %% Iterate between the update steps until convergence of max number of iterations 
    for counter = 1:MAX_ITER
        
        % 1) Fix alpha, B1, B2 and update X 
        
        r = fun_b(Y,alpha) - fun_G(Mu,alpha,B1,B2); 
        for l=1:GRAD_DESC_STEPS 
            % prepare 
            R = reshape(r, n,n); 
            Gr = fun_G(R,alpha,B1,B2); 
            % update rules 
            gamma = norm(r)^2/( dot(r,Gr) ); % step size  
            Mu = Mu + gamma*R; % update 
            r = r - gamma*Gr;          
        end
        
        % 2) Fix x, B and update alpha 
        alpha = ( m^2 + 2*c )./( norm( vec( F_1d*Mu*(F_1d')- Y ) )^2 + 2*d );   
        
        % 3) Fix x, alpha and update B1 and B2  
        B1 = (1+2*c)./( (D*Mu).^2 + 2*d ); 
        B2 = (1+2*c)./( (Mu*(D')).^2 + 2*d ); 
         
        % store certain values in history structure 
        history.abs_error(counter) = norm( vec( Mu-Mu_OLD ) )^2;%/length(vec(Mu)); % absolute error 
        history.rel_error(counter) = ( norm( vec( Mu-Mu_OLD ) )/norm( vec( Mu_OLD ) ) )^2; % relative error        
        Mu_OLD = Mu; % store value of mu 
        
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