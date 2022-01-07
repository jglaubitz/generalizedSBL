%% script_MRI_2d
%
% Description: 
%  Script to reconstruct images based on noisy incomplete discrete Fourier data. 
%
% Author: Jan Glaubitz 
% Date: Jan 07, 2022
%

clear all; close all; clc; % clean up
%warning('off','all') % in case any of the warnings become too anoying 


%% Free parameters 

% Free parameters
n = 200; % number of pixels in every direction 
nr_rmvd = 100; % number of frquencies/coefficients that are removed 
sampling = 'log'; % sampling strategy (quadratic, log)
noise_variance = 10^(-3); % variance of the i.i.d. complex Gaussian noise added to the measurements
order = 1; % order of the TV/PA operator (1,2,3) 
c = 1; d = 10^(-2); % hyper-hyper-parameters

%% Set up the model 

% Test image 
X = phantom(n);; % set up test image 
RI = imref2d(size(X)); 
x = X(:); % vectorize the image by stacking up the columns 

% remove certain samples 
if strcmp( sampling, 'quadratic') 
    samples_rmvd = ( ( 2:sqrt(n) ).^2 )'; 
elseif strcmp( sampling, 'log')     
    samples_rmvd = floor(logspace(1,log10(n),nr_rmvd));
else 
    error('Type of sampling not yet implemented') 
end 

% Data model and noise 
F_1d_complex = dftmtx(n)/sqrt(n); % matrix corresponding to the one-dimensional normalized DFT 
%F_1d_undersampled = F_1d_complex(samples,:); % only keep a smaller number of rows 
F_1d_complex(samples_rmvd,:) = [];
F_1d = [real(F_1d_complex); imag(F_1d_complex)]; % real-valued forward operator
rng('default'); rng(1,'twister'); % to make the results reproducable
noise = sqrt(noise_variance/2)*randn(size(F_1d,1),size(F_1d,1)); % iid real Gaussian noise 
Y = F_1d*X*(F_1d') + noise; % real-valued noisy indirect measuremnt 
y = Y(:); 

% Regularization operator  
D = TV_operator( n, order ); % regularization operator 


%% Use different methods for reconstruction 

% ML estimate 
lambda = 10^(-12); % almost no regulairzation 
rho = 1.0; alpha = 1.0; % ADMM parameters 
[X_LS, history_LS] = ADMM_2d(F_1d, Y, D, lambda, rho, alpha, 0);

% SBL based on Bayesian coordinate descent 
[Mu, alpha, B1, B2, history] = BCD_2d( F_1d, Y, D, c, d );
X_BCD = Mu; 

% ADMM 
lambda = 4*noise_variance; % Parameter selection for ADMM 
rho = 1.0; alpha = 1.0; % ADMM parameters 
[X_l1, history_l1] = ADMM_2d(F_1d, Y, D, lambda, rho, alpha, 0); 

% Compute SNR 
SNR = norm(x)^2/(length(x)*noise_variance) 


%% Plot the results 

% Exact image 
f = figure(1);
ax = axes(f);
imshow(X, RI, 'InitialMagnification',400, 'Parent',ax); 
colorbar;
set(gca, 'FontSize', 18); % Increasing ticks fontsize 

% Noisy blurred image 
f = figure(2);
ax = axes(f);
imshow(X_LS, RI, 'InitialMagnification',400, 'Parent',ax); 
colorbar;
set(gca, 'FontSize', 18); % Increasing ticks fontsize 

% l1-reg by ADMM 
f = figure(3);
ax = axes(f);
imshow(X_l1, RI, 'InitialMagnification',400, 'Parent',ax); 
colorbar;
set(gca, 'FontSize', 18); % Increasing ticks fontsize 

% SBL by BCD 
f = figure(4);
ax = axes(f);
imshow(X_BCD, RI, 'InitialMagnification',400, 'Parent',ax); 
colorbar;
set(gca, 'FontSize', 18); % Increasing ticks fontsize 