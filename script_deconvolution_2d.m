%% script_deconvolution_2d
%
% Description: 
%  Script to reconstruct a natural image based on noisy blurred data. 
%
% Author: Jan Glaubitz 
% Date: Jan 07, 2022
%

clear all; close all; clc; % clean up
%warning('off','all') % in case any of the warnings become too anoying 


%% Free parameters 

% Free parameters
n = 125; % number of pixels in every direction 
gamma = 0.015; % blurring parameter (Gaussian convolution kernel) 
noise_variance = 10^(-5); % variance of the i.i.d. complex Gaussian noise added to the measurements 
order = 2; % order of the TV/PA operator (1,2,3) 
c = 1; d = 10^(-2); % hyper-hyper-parameters


%% Set up the model 

% Test image 
I = imread('satellite.png');
I = im2double(I);
X = imresize(I,[n n]);
RI = imref2d(size(X)); 
x = X(:); % vectorize the image by stacking up the columns 

% forward operator, noise, and data 
F_1d = construct_F_deconvolution( n, gamma ); % 1d forward operator 
rng('default'); rng(1,'twister'); % to make the results reproducable 
noise = sqrt(noise_variance/2)*randn(size(F_1d,1),size(F_1d,1)); % iid real Gaussian noise 
Y = F_1d*X*(F_1d') + noise; % real-valued noisy indirect measuremnt 
y = Y(:); 

% Regularization operator  
D = TV_operator( n, order ); 


%% Use different methods for reconstruction 

% SBL based on Bayesian coordinate descent 
[Mu, alpha, B1, B2, history] = BCD_2d( F_1d, Y, D, c, d );
X_BCD = Mu; 

% ADMM 
lambda = noise_variance; % Parameter selection for ADMM 
rho = 1.0; alpha = 1.0; % ADMM parameters 
[X_l1, history_l1] = ADMM_2d(F_1d, Y, D, lambda, rho, alpha, 0); 

% Compute SNR 
SNR = norm(x)^2/(length(x)*noise_variance)  


%% Plot the results 

% Exact image 
f = figure(1);
ax = axes(f);
imshow(1-X, RI, 'InitialMagnification',600, 'Parent',ax); 
colorbar;
set(gca, 'FontSize', 18); % Increasing ticks fontsize 

% Noisy blurred image 
f = figure(2);
ax = axes(f);
imshow(1-Y, RI, 'InitialMagnification',600, 'Parent',ax); 
colorbar;
set(gca, 'FontSize', 18); % Increasing ticks fontsize 

% l1-reg by ADMM 
f = figure(3);
ax = axes(f);
imshow(1-X_l1, RI, 'InitialMagnification',600, 'Parent',ax); 
colorbar;
set(gca, 'FontSize', 18); % Increasing ticks fontsize 

% SBL by BCD 
f = figure(4);
ax = axes(f);
imshow(1-X_BCD, RI, 'InitialMagnification',600, 'Parent',ax); 
colorbar;
set(gca, 'FontSize', 18); % Increasing ticks fontsize 