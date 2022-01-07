%% script_deconvolution_1d
%
% Description: 
%  Script to reconstruct a piecewise constant signal based on noisy blurred data. 
%
% Author: Jan Glaubitz 
% Date: Jan 07, 2022
%

clear all; close all; clc; % clean up
%warning('off','all') % in case any of the warnings become too anoying 


%% Free parameters 

% Free parameters of the problem 
n = 40; % number of (equidistant) data points on [0,1] 
gamma = 0.03; % blurring parameter (Gaussian convolution kernel)
noise_variance = 0.01; % variance of the iid Gaussian noise added to the measurements

% Free parameters of SBL 
c = 1; d = 10^(-4); % hyper-hyper-parameters


%% Set up the model 

% Test function 
fun = @(t) (t<0.15).*(-1) + (t>=0.15 & t<0.25 ).*(0) + ... 
    (t>=0.25 & t<0.5 ).*(1) + (t>=0.5 & t<0.75 ).*(-0.5) + ... 
    (t>=0.75 & t<0.85 ).*(1.75) + (t>=0.85).*(0.5);

% Data points and signal values 
data_points = linspace(0, 1, n)'; % equidistant data points 
x = fun(data_points); % function values at grid points 

% forward operator, noise, and data 
F = construct_F_deconvolution( n, gamma ); 
rng('default'); rng(1,'twister'); % to make the results reproducable 
noise = sqrt(noise_variance/2)*randn(n,1); % iid normal noise
y = F*x + noise; % noisy indirect data 

% Regularization operator 
order = 1; 
R = TV_operator( n, order ); % regularization operator 


%% Use different methods for reconstruction 

% SBL based on Bayesian coordinate descent 
[x_BCD, C_inv, alpha, beta, history] = BCD_1d( F, y, R, c, d, 0 ); 

% ADMM 
nr_jumps = 5; 
lambda = 2*noise_variance*nr_jumps; % Parameter selection for ADMM 
rho = 1.0; alpha = 1.0; % ADMM parameters 
[x_l1, history_l1] = ADMM_1d(F, y, R, lambda, rho, alpha, 0);

% Compute SNR 
SNR = norm(x)^2/(n*noise_variance)  


%% Compute lower and upper bounds of the confidence intervals 
C = inv(C_inv); % get the covariance matrix 
[CI_lower, CI_upper] = compute_CI( x_BCD, C ); % computer lower and upper bounds


%% Plot the results 

% Exact solution and measurements 
figure(1) 
p1 = fplot( fun, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, y, 'ro' ); 
set(p1, 'LineWidth',3);
set(p2, 'markersize',10, 'LineWidth',2.5); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
xlabel('$t$','Interpreter','latex'); 
ylabel('$x$','Interpreter','latex'); 
grid on 
lgnd = legend('true signal $x$','noisy blurred data $\mathbf{y}$'); 
%lgnd = legend('$x$','$\mathbf{y}$');  
set(lgnd, 'Interpreter','latex', 'FontSize',22, 'color','none', 'Location','best')
hold off

% Reconstructions 
figure(2) 
p1 = fplot( fun, [0,1], 'k:'); % plot the reference solution 
hold on
p2 = plot( data_points, y, 'ro', data_points, x_l1, 'g^', data_points, x_BCD, 'bs'); % reconstructions 
set(p2(2), 'color', [0 0.75 0])
set(p1, 'LineWidth',3); 
set(p2, 'markersize',10, 'LineWidth',2); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
xlabel('$t$','Interpreter','latex'); 
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, '$\mathbf{y}$', 'ADMM', 'BCD');
set(lgnd, 'Interpreter','latex', 'FontSize',22, 'color','none', 'Location','best')
hold off 

% Plot varying regularization parameter beta  
figure(3) 
p1 = fplot( fun, [0,1], 'k:'); % plot the reference solution 
hold on 
eval_points = (data_points(1:end-1)+data_points(2:end))/2; % mid points 
beta_inv = 1./beta; 
p2 = plot( eval_points, beta_inv/max(beta_inv), 'r--' ); 
set(p1, 'LineWidth',3); 
set(p2, 'LineWidth',3); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
xlabel('$t$','Interpreter','latex'); 
%axis([0 1 -0.1 2.1])
grid on 
lgnd = legend('true signal $x$','normalized $\beta^{-1}$', 'Interpreter','latex','Location','northwest');  
set(lgnd, 'Interpreter','latex', 'FontSize',22, 'color','none')
hold off 

% Plot confidence intervals 
figure(4) 
p1 = fplot( fun, [0,1], 'k:'); % plot the reference solution 
set(p1, 'LineWidth',3); 
hold on 
p2 = plot( data_points, x_BCD, 'bs' ); % reconstructions 
set(p2, 'markersize',10, 'LineWidth',2); 
p3 = patch([data_points; flipud(data_points)], [CI_lower; flipud(CI_upper)], [1 0.4 0]);  
set(p3, 'EdgeColor','none', 'FaceAlpha',0.35); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
xlabel('$t$','Interpreter','latex'); 
ylabel('$x$','Interpreter','latex'); 
grid on 
%axis([0 1 -1 2.5])
lgnd = legend([p2 p3], 'mean','$99.9\%$ CI', 'Interpreter','latex','Location','northwest');  
set(lgnd, 'Interpreter','latex', 'FontSize',22, 'color','none')
hold off 