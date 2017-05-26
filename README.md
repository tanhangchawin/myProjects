# myProjects
# Linear regression with validation curve and regularization 
# Octave/Matlab

function [theta, J_history] = main(allX, allY) %this is my main function
  
  %randomly initialize the training examples
  Num = size(allX, 1);
  rand_indices = randperm(Num);
  
  
  %split them up into training and validation sets
  xdata = allX(rand_indices(1:round(0.6*Num)), :);
  xval = allX(rand_indices(round(0.6*Num+1):round(0.8*Num)), :);
  y = allY(rand_indices(1:round(0.6*Num)), :);
  yval = allY(rand_indices(round(0.6*Num+1):round(0.8*Num)), :);
  
  m = length(y);
  n = size(xdata, 2);
  
  %initialize lambda
  lambda = 1.5;
  
  %add a column of ones in front of X
  X = [ones(m, 1), xdata];
  
  % log the y values
  [yLog, yValLog] = dataLog(y, yval);
  
  %initialize theta
  theta = zeros(n+1, 1);
  
  %initialize alpha and number of iterations here
  alpha = 0.01;
  num_iters = 200;
  
  %run gradient descent
  [thetaFinal, J_history] = gradientDescent(X, yLog, theta, alpha, num_iters, lambda)
  
  %compute and plot validation error
  [errorTrain, errorVal] = mainVal(X, yLog, xval, yValLog, theta, alpha, num_iters, lambda);
  
  %plot validation error
  plot(1:m, errorTrain, 1:m, errorVal);
  title('Validation curve');
  xlabel('Number of Training sets');
  ylabel(' Error');
  legend('Trained','Cross-validation');
  

function J = computeCost(X, yLog, theta, lambda) %function to compute cost
  
  m = length(yLog); % number of training example
  J = 0;
  J = sum(((X * theta) - yLog) .^ 2);
  J = 1 / (2 * m) * J + (lambda/(2*m))*sum(theta.^2);
  
function [theta, J_history] = gradientDescent(X, yLog, theta, alpha, num_iters, lambda) % gradient descent
  
  m = length(yLog);
  J_history = zeros(num_iters, 1);
  
  for iter = 1:num_iters
    A = X*theta - yLog;
    %compute theta
    delta = 1 / m * (A' * X)'.+(lambda/m)*[0; theta(2:length(theta))];  % ' ((n+1) x 1 vector), similar to theta, add regularization term
    theta = theta - (alpha * delta); % ' ((n+1) x 1 vector)
    %compute cost hisotry
    cost = computeCost(X, yLog, theta, lambda);
	  J_history(iter) = cost;
  end

function [errorTrain, errorVal] = mainVal(X, yLog, xval, yvalLog, theta, alpha, num_iters, lambda) % compute validation error
   
   % this function determines high bias or high variance
   m = length(yLog);
   mXval = length(yvalLog);%number of validation set
   
   errorTrain = zeros(m,1);
   errorVal = zeros(m, 1);
   
   Xval = [ones(mXval, 1), xval];
   
   for i = 1:m
     theta = gradientDescent(X(1:i, :), yLog(1:i), theta, alpha, num_iters, lambda);
     errorTrain(i) = 1 / (2 * i) * sum(((X(1:i, :) * theta) - yLog(1:i)) .^ 2);
     errorVal(i) = 1 / (2 * size(yvalLog, 1)) * sum(((Xval * theta) - yvalLog) .^ 2);
   end
   
function [yLog, yValLog] = dataLog(y, yval)
  
  %initialize yLog and yValLog
  
  m = size(y, 1);
  n = size(yval, 1);
  
  yLog = zeros(m, 1);
  yValLog = zeros(n, 1);
   
  for i = 1:m
    if (y(i) == 0)
      yLog(i) = 0;
    else
      yLog(i) = log(y(i));
    end
  end 
  
  for j = 1:n
    if (yval(j) == 0)
      yValLog(j) = 0;
    else
      yValLog(j) = log(yval(j));
    end
  end
