%% compute_CI
%
% Description: 
%  Function that computes the lower and upper bound of the confidence
%  intervals corresponding to the different components of the solution
%  posterior 
% 
% INPUT: 
%  mu :     vector of means 
%  C :      covariance matrix
%
% OUTPUT: 
%  CI_lower :   lower bounds of the confidence intervals 
%  CI_upper :   upper bounds of the confidence intervals 
%
% Author: Jan Glaubitz 
% Date: Jan 07, 2022
%

function [CI_lower, CI_upper] = compute_CI( mu, C )

    N = 1000; % number of samples 
    n = length(mu); % number of random variables 
    CI_lower = zeros(n,1); CI_upper = zeros(n,1); 
    
    % Sample from the multivariate normal distribution 
    rng default % For reproducibility 
    Samples = mvnrnd(mu,C,N); % sample
    
    for i=1:n 
       
        y = Samples(:,i); % look at the i-th component of the samples 
        % calculate 99% confidence intervals 
        CI_lower(i) = quantile(y,0.001); 
        CI_upper(i) = quantile(y,0.999);  
        
    end
    
end