function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
M = size(X, 2);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%% --- feedforwarding the NN

% input layer
% adding bias to the entering vector
a1 = [ones(1,M); X];

% hidden layer
z2 = Theta1 * a1;
a2 = [ones(1,M); sigmoid(z2)]; % adds bias

% output layer
z3 = Theta2 * a2;
a3 = sigmoid(z3);

% hipothesys
h_x = a3;

% -- Computing the cost

% first part of the cost function
J1 = 0;
for m = 1:M         % iterates for all training examples

        % cost function without regularization and factor -1/m
        J1 = J1 + (-y(:,m).'*log(h_x(:,m)) - ((1-y(:,m)).'*log(1-h_x(:,m))));
    
end
J1 = (1/M) * J1;

% second part: regularization cost 
J2 = (lambda/(2*M)) * sum(nn_params.^2);

% -- Final cost with regularization
J = J1 +J2;

%% --- Backprop

% computing delta matrices
for m=1:M
    
    % layer 3 (last) error
    delta_3 = a3(:,m) - y(:,m);
    
    % layer 2 error
    aux_delta_2 = (Theta2.' * delta_3);
    delta_2 = aux_delta_2(2:end) .*  sigmoidGradient(z2(:,m)); % eliminates first row of aux_delta_2 because of bias
    
    % deltas
    Delta2 = Delta2 + delta_3 * a2(:,m).';
    
    Delta1 = Delta1 + delta_2 * a1(:, m).';
    
end

% regularized gradient for the neural network cost function
Theta1_grad = (1/M) * Delta1 + (lambda/M) * Theta1;
Theta1_grad(:,1) = (1/M) * Delta1(:,1);

Theta2_grad = (1/M) * Delta2 + (lambda/M) * Theta2;
Theta2_grad(:,1) = (1/M) * Delta2(:,1);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
