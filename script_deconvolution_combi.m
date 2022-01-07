%% script_deconvolution_combi
%
% Description: 
%  Script to reconstruct a piecewise constant and linear signal based on noisy blurred data. 
%  This is supposed to demonstrate the potential advantage of combining
%  different regularization operators. 
%
% Author: Jan Glaubitz 
% Date: Jan 07, 2022
%

clear all; close all; clc; % clean up
%warning('off','all') % in case any of the warnings become too anoying 


%% Free parameters 

% Free parameters of the problem 
n = 40; % number of (equidistant) data points on [0,1] 
gamma = 0.01; % blurring parameter (Gaussian convolution kernel)
noise_variance = 0.01; % variance of the iid Gaussian noise added to the measurements

% Free parameters of SBL 
c = 1; d = 10^(-4); % hyper-hyper-parameters


%% Set up the model 

% Test function 
fun = @(t) (t<0.25).*(0) + (t>=0.25 & t<0.5 ).*(1) + ... 
    (t>=0.5 & t<0.75 ).*(-4*(t-0.5)) + (t>=0.75).*(1-4*(t-0.75));

% Data points and signal values 
data_points = linspace(0, 1, n)'; % equidistant data points 
x = fun(data_points); % function values at grid points 

% forward operator, noise, and data 
F = construct_F_deconvolution( n, gamma ); 
rng('default'); rng(1,'twister'); % to make the results reproducable 
noise = sqrt(noise_variance/2)*randn(n,1); % iid normal noise
y = F*x + noise; % noisy indirect data 


%% Reconstructions with different regaurlization operators 

% First-order TV
R = TV_operator( n, 1 ); % Regularization operator 
% SBL based on Bayesian coordinate descent 
[x_BCD_TV1, C_inv, alpha, beta, history] = BCD_1d( F, y, R, c, d, 0 );

% Second-order TV
R = TV_operator( n, 2 ); % Regularization operator 
% SBL based on Bayesian coordinate descent 
[x_BCD_TV2, C_inv, alpha, beta, history] = BCD_1d( F, y, R, c, d, 0 );

% First- and second-order combined
R1 = TV_operator( n/2, 1 ); 
R2 = TV_operator( n/2, 2 ); 
R = [R1,zeros(n/2-1,n/2);zeros(n/2-2,n/2),R2]
% SBL based on Bayesian coordinate descent 
[x_BCD_combi, C_inv, alpha, beta, history] = BCD_1d( F, y, R, c, d, 0 );

% Compute SNRs 
SNR = norm(x)^2/(length(x)*noise_variance) 


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
p2 = plot( data_points, x_BCD_TV1, 'g^', data_points, x_BCD_TV2, 'bs', data_points, x_BCD_combi, 'r*'); % reconstructions 
set(p2(1), 'color', [0 0.75 0])
set(p1, 'LineWidth',3); 
set(p2, 'markersize',10, 'LineWidth',2); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
xlabel('$t$','Interpreter','latex'); 
ylabel('$x$','Interpreter','latex'); 
grid on  
lgnd = legend(p2, '$1$st-order TV', '$2$nd-order TV', 'combined');
set(lgnd, 'Interpreter','latex', 'FontSize',22, 'color','none', 'Location','best')
hold off 