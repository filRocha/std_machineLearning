function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% computing the hypothesis. Do you remember what the hypothesis is???
h = X * theta;

% The unregularized code
j1 = (1/(2*m)) * sum((h - y).^2); 

% the regularization term
j2 = (lambda/(2*m)) * sum(theta(2)^2);

% summing them up -> this is the regularized cost for the given theta
J = j1 + j2;


% -- computing the gradient
% one may see that the first gradient has a different formula
grad(1) = (1/m) * sum((h - y) .* X(:,1));

grad(2) = ((1/m) * sum((h - y) .* X(:,2))) + ((lambda/m)*theta(2));



% =========================================================================

grad = grad(:);

end
