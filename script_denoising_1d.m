%% script_denoising_1d
%
% Description: 
%  Script to reconstruct a sparse signal based on noisy observations. 
%
% Author: Jan Glaubitz 
% Date: Jan 07, 2022 
%

clear all; close all; clc; % clean up
%warning('off','all') % in case any of the warnings become too anoying 


%% Free parameters 

% Free parameters
n = 20; % number of (equidistant) data points on [0,1] 
n_support = 4; % number of nonzero values 
noise_variance = 0.05; % variance of the iid Gaussian noise added to the measurements
c = 1; d = 10^(-4); % hyper-hyper-parameters for BCD
c_IAS = 1.55; d_IAS = 0.05; % hyper-hyper-parameters for IAS

%% Set up the model 

% Data points and signal values 
data_points = linspace(0, 1, n)'; % equidistant data points 
rng('default'); rng(1,'twister'); % to make the results reproducable 
index = randperm(n,n_support)'; % randomly select intergers between 1 and n 
index = sort(index); % sort them increasingly 
x = zeros(n,1); % zero values 
x(index) = 1; % nonzero values (scalled such that largest value is 1)

% forward operator, noise, and data 
F = speye(n); % denoising (F=I) 
noise = sqrt(noise_variance/2)*randn(n,1); % iid normal noise
y = F*x + noise; % noisy indirect data 

% Regularization operator 
R = speye(n); % regularization operator 


%% Use different methods for reconstruction 

% SBL based on Bayesian coordinate descent 
[x_BCD, C_inv, alpha, beta, history] = BCD_1d( F, y, R, c, d, 0 ); 

% SBL - Evidence approach 
[x_evidence, C_inv, alpha, beta, history] = SBL_evidence_1d( F, y, c, d, 0 ); 

% SBL - IAS 
[x_IAS, beta, history] = IAS_1d( F, y, noise_variance, c_IAS, d_IAS, 0 );  

% ADMM 
lambda = 2*noise_variance*n_support; % Parameter selection for ADMM 
rho = 1.0; alpha = 1.0; % ADMM parameters 
[x_l1, history_l1] = ADMM_1d(F, y, R, lambda, rho, alpha, 0);

% Compute SNR 
SNR = norm(x)^2/(n*noise_variance)  


%% Plot the results 

% Exact solution and measurements 
figure(1) 
p1 = plot( data_points, x, 'k:' ); 
hold on
p2 = plot( data_points, y, 'ro' ); 
set(p1, 'LineWidth',3);
set(p2, 'markersize',10, 'LineWidth',2.5); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
xlabel('$t$','Interpreter','latex'); 
ylabel('$x$','Interpreter','latex'); 
grid on 
%lgnd = legend('true signal $x$','noisy data $\mathbf{y}$'); 
lgnd = legend('$x$','$\mathbf{y}$');  
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location','best')
hold off

% Reconstructions 
figure(2) 
p1 = plot( data_points, x, 'k:' );  
hold on
p2 = plot( data_points, x_BCD, 'bs', data_points, x_evidence, 'g^', data_points, x_IAS, 'mo', data_points, x_l1, 'r*' ); 
set(p2(2), 'color', [0 0.75 0])
set(p1, 'LineWidth',3); 
set(p2, 'markersize',10, 'LineWidth',2); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
xlabel('$t$','Interpreter','latex'); 
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, 'BCD', 'evidence', 'IAS', 'ADMM');
set(lgnd, 'Interpreter','latex', 'FontSize',22, 'color','none', 'Location','best')
hold off 