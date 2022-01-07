%% construct_F_deconvolution
%
% Description: 
%  Function that constructs the forward operator corresponding to 
%  convolution using a Gaussian kernel with parameter gamma 
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

function F = construct_F_deconvolution( n, gamma )

    kernel = @(t) exp( -t.^2/(2*gamma^2) )/sqrt(2*pi*gamma^2); % kernel 
    grid = linspace(0, 1, n); % equidistant grid points
    F = kernel( grid-grid' )/n;
    
end