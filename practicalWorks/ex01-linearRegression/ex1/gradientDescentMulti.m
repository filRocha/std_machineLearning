function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % Iterates 4all training examples

    % Computes the error for each x 
    aux = theta.' * X.' - y.';
    
    for i=1:size(theta,1)
        dJ(i,:) = sum(aux.' .* X(:,i));
    end
    
    % Updating theta
    theta = theta - alpha * (1/m) * dJ;
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end



end
