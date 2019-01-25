function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 2);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 2), 1);

z1 = [ones(1, m); X].';
h1 = sigmoid( z1 * Theta1.');

z2 = [ones(1, m).', h1];
h2 = sigmoid( z2 * Theta2.');

for i=1:m
   if h2(i) > 0.5
    p(i) = 1;
   else
    p(i) = 0;
end



% =========================================================================


end
