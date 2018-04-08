%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STOCHASTIC PROGRAMMING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% the purpose of this project is to implement a two-stage stochastic linear
% program which solves a tuition investment-matching strategy.

% authors: Matthew Reiter and Daniel Kecman
% date: april 11, 2018

clc
clear all
format long

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. read input files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load the stock weekly prices and factors weekly returns
data = readtable('price_data.csv');
data.Properties.RowNames = cellstr(datetime(data.Date));
data = data(:,2:size(data,2));

% n represents the number of stocks that we have
n = size(data,2);

% identify the tickers and the dates 
tickers = data.Properties.VariableNames';
dates = datetime(data.Properties.RowNames);

% calculate the stocks' weekly returns
prices  = table2array(data);
returns = (prices(2:end,:) - prices(1:end-1,:)) ./ prices(1:end-1,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. estimation of mean and variance
%   - we use historial data spanning a year from 2014-01-03 to 2014-12-26
%   - we use the geometric mean for stock returns and from this we formulate
%   the covariance matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% calibration start data and end data
cal_start = datetime('2014-01-03');
cal_end = cal_start + calmonths(12) - days(2);

% testing period start date and end date
test_start = cal_end + days(1);
test_end = test_start + calmonths(12) - days(2);

cal_returns = returns(cal_start <= dates & dates <= cal_end,:);
current_prices = table2array(data((cal_end - days(7)) <= dates & dates <= cal_end,:))';

% calculate the geometric mean of the returns of all assets
mu = (geomean(cal_returns+1)-1)';
cov = cov(cal_returns);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. scenario generation
%   - we use two methods to generate scenarios, seperate for the scenario's
%   governing the asset returns and for the matched liabilities
%   - for asset returns, we model the stock price as a Geometric Brownian
%   Motion and do a Monte Carlo simulation to estimate stock returns for
%   our investment period spanning a year from 2015-01-02 to 2015-12-31
%   - for our liabilties, we sample from a normal distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% we need the correlation matrix to simulate the correlated prices of 
% portfolio
rho = corrcov(cov);

% we take the cholesky factorization of the correlation matrix
L = chol(rho, 'lower');

% define the number of randomized scenarios to sample for
S = 20;

% our simulated asset prices and returns
sim_price = zeros(n,S);
sim_returns = zeros(n,S);

% our scenario liabilities
sim_liabilities = zeros(S,1);

% we have yearly estimates for returns and we wish to simulate the
% price path after six months using monthly time-steps
dt = 1/12;

% setting the random seed so that our random draws are consistent across testing
% rng(1);

for i=1:S

    % our random correlated pertubations
    epsilon = L * normrnd(0,1,[n,1]);
    
    % randomize our liabilities
    sim_liabilities(i) = 17000 + normrnd(500,200);

    % calculate our simulated prices
    sim_price(:,i) = current_prices .* exp((mu - 0.5 * diag(cov))*dt + sqrt(dt)*sqrt(diag(cov)) .* epsilon);  

    % calculate our simulated returns
    sim_returns(:,i) = (sim_price(:,i) - current_prices) ./ current_prices;
end

X = 1:n;
Y = 1:S;
mesh(sim_price);
title('Simulated Prices of Holding Assets', 'FontSize', 14)
ylabel('Asset','interpreter','latex','FontSize',12);
xlabel('Scenario','interpreter','latex','FontSize',12);
zlabel('Asset Price','interpreter','latex','FontSize',12);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. stochastic optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% take uniform probability for each scenario
p = 1/S;

% we have an initial budget of 15,000
B = 12000;

% the benefit of running a surplus will be a 1 and the cost of running a
% shortfall will be -10
surplus = -2;
shortfall = 10;

% formulate our objective function
% NOTE: our variables are organized as follows
f = [ones(1,n) repmat(p*[surplus shortfall], 1, S)]';

% dealing with our first-stage constraints
A = [ones(1,n) zeros(1,2*S)];
b = B;

% handling our second stage constraints
temp = [-1 1];
temp_r = repmat(temp, 1, S);
temp_c = mat2cell(temp_r, size(temp,1), repmat(size(temp,2),1,S));
con = blkdiag(temp_c{:});

Aeq = [(sim_returns+1)' con];
beq = sim_liabilities;

lb = zeros(n+2*S,1);
ub = [];

[optimal, value] = linprog(f, A, b, Aeq, beq, lb, ub)









