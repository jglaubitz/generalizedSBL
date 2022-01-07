%% script_dataFusion_1d
%
% Description: 
% Script to reconstruct a (one-dimensional) signal based on data fusion.
%
% Author: Jan Glaubitz 
% Date: Jan 07, 2022
%

clear all; close all; clc; % clean up
%warning('off','all') % in case any of the warnings become too anoying 


%% Free parameters 

% Free parameters of the problem 
n = 40; % number of (equidistant) data points on [0,1] 
noise_variance1 = 0.5; % variance of the first sensor 
noise_variance2 = 0.01; % variance of the second sensor 
gamma = 0.03; % blurring parameter (Gaussian convolution kernel)
ratio_rmvd1 = 0.1; % percantage of coefficients that are removed from first sensore
ratio_rmvd2 = 0.4; % percantage of coefficients that are removed from second sensore

% Free parameters of SBL 
c = 1; d = 10^(-4); % hyper-hyper-parameters


%% Set up the model 

rng('default'); rng(1,'twister'); % to make the results reproducable 

% Test function 
fun = @(t) (t<0.15).*(-1) + (t>=0.15 & t<0.25 ).*(0) + ... 
    (t>=0.25 & t<0.5 ).*(1) + (t>=0.5 & t<0.75 ).*(-0.5) + ... 
    (t>=0.75 & t<0.85 ).*(1.75) + (t>=0.85).*(0.5);

% Data points and signal values 
data_points = linspace(0, 1, n)'; % equidistant data points 
x = fun(data_points); % function values at grid points 

% forward operator, noise, and data - 1st sensor 
F1 = speye(n); % denoising (F=I) 
nr_rmvd = ceil(n*ratio_rmvd1); % number of removed frquencies 
index = randperm(n,nr_rmvd)'; % randomly select intergers between 1 and n 
index = sort(index); % sort them increasingly 
F1(index,:) = []; % remove rows from forward operator 
data_points1 = data_points; 
data_points1(index,:) = []; % also remove corresponding points 
m1 = size(F1,1); % number of measurements 
noise1 = sqrt(noise_variance1/2)*randn(m1,1); % iid normal noise
y1 = F1*x + noise1; % noisy indirect data 

% forward operator, noise, and data - 2nd sensor 
F2 = construct_F_deconvolution( n, gamma ); % convolution 
nr_rmvd = ceil(n*ratio_rmvd2); % number of removed frquencies 
index = randperm(n,nr_rmvd)'; % randomly select intergers between 1 and n 
index = sort(index); % sort them increasingly 
F2(index,:) = []; % remove rows from forward operator 
data_points2 = data_points; 
data_points2(index,:) = []; % also remove corresponding points
m2 = size(F2,1); % number of measurements 
noise2 = sqrt(noise_variance2/2)*randn(m2,1); % iid normal noise
y2 = F2*x + noise2; % noisy indirect data 

% Combine data models 
F = [F1;F2]; % combined forward operator 
y = [y1;y2]; 

% Regularization operator 
order = 1; 
R = TV_operator( n, order ); % regularization operator 


%% Use different methods for reconstruction 

% BCD - only 1st sensor  
[x_BCD_sensor1, C_inv, alpha, beta, history] = BCD_1d( F1, y1, R, c, d, 0 ); 

% BCD - only 2nd sensor  
[x_BCD_sensor2, C_inv, alpha, beta, history] = BCD_1d( F2, y2, R, c, d, 0 ); 

% BCD - combined but iid assumption 
F = [F1;F2]; y = [y1;y2]; % combined forward operator and data 
[x_BCD_comb_iid, C_inv, alpha, beta, history] = BCD_1d( F, y, R, c, d, 0 ); 

% BCD = combined and general noise model 
[x_BCD_comb, C_inv_fusion, alpha, beta, history] = BCD_1d_fusion( F1, F2, y1, y2, R, c, d, 0 );

% Compute SNRs 
SNR1 = norm(x)^2/(length(x)*noise_variance1) 
SNR2 = norm(x)^2/(length(x)*noise_variance2) 


%% Plot the results 

% 1st sensor 
figure(1) 
p1 = plot( data_points, x, 'k:' ); 
hold on
p2 = plot( data_points1, y1, 'ro', data_points, x_BCD_sensor1, 'bs' ); 
set(p1, 'LineWidth',3);
set(p2, 'markersize',10, 'LineWidth',2.5); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
xlabel('$t$','Interpreter','latex'); 
ylabel('$x$','Interpreter','latex'); 
grid on 
lgnd = legend('true signal','noisy data','BCD');  
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location','best')
hold off

% 2nd sensor 
figure(2) 
p1 = plot( data_points, x, 'k:' ); 
hold on
p2 = plot( data_points2, y2, 'g^', data_points, x_BCD_sensor2, 'bs' ); 
set(p2(1), 'color', [0 0.75 0])
set(p1, 'LineWidth',3);
set(p2, 'markersize',10, 'LineWidth',2.5); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
xlabel('$t$','Interpreter','latex'); 
ylabel('$x$','Interpreter','latex'); 
grid on 
lgnd = legend('true signal','noisy blurred data','BCD');  
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location','best')
hold off

% Combined - iid assumption 
figure(3) 
p1 = plot( data_points, x, 'k:' ); 
hold on
p2 = plot( data_points1, y1, 'ro', data_points2, y2, 'g^', data_points, x_BCD_comb_iid, 'bs' ); 
set(p2(2), 'color', [0 0.75 0])
set(p1, 'LineWidth',3);
set(p2, 'markersize',10, 'LineWidth',2.5); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
xlabel('$t$','Interpreter','latex'); 
ylabel('$x$','Interpreter','latex'); 
grid on 
lgnd = legend(p2, 'noisy data','noisy blurred data','BCD');  
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location','best')
hold off

% Combined - no iid assumption 
figure(4) 
p1 = plot( data_points, x, 'k:' ); 
hold on
p2 = plot( data_points1, y1, 'ro', data_points2, y2, 'g^', data_points, x_BCD_comb, 'bs' ); 
set(p2(2), 'color', [0 0.75 0])
set(p1, 'LineWidth',3);
set(p2, 'markersize',10, 'LineWidth',2.5); 
set(gca, 'FontSize', 24); % Increasing ticks fontsize 
xlabel('$t$','Interpreter','latex'); 
ylabel('$x$','Interpreter','latex'); 
grid on 
lgnd = legend(p2, 'noisy data','noisy blurred data','BCD');  
set(lgnd, 'Interpreter','latex', 'FontSize',26, 'color','none', 'Location','best')
hold off