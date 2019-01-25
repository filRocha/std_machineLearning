% code renatilda
clear all;
clc;
close all;
fprintf('\n --- code started -- \n');

%% parameters

% lambda parameters
lambda = 0;

% nn architecture
% nn parameters
layer_size_input = 4;
layer_size_hidden1 = 20;
labels_number = 1;

% number of training iterations
training_iterations = 3000;

%% preparing data

% loading the data
data = xlsread('escor.xlsx','ESCOR');

% saving in the vector
X = data(:, 1:4).';
y = data(:, 5).';

% training examples count
M = size(X,2);

% randomnly initialize parameters
initial_Theta1 = randInitializeWeights(layer_size_input, layer_size_hidden1);
initial_Theta2 = randInitializeWeights(layer_size_hidden1, labels_number);

% unrolled parameters
initial_nn_params = [initial_Theta1(:); initial_Theta2(:)];

%% training the neural net

% defined parameter
options = optimset('MaxIter',training_iterations);

 % remember that, if y is not binary vectorized, you should do it first!
% create a handler for the cost function
costFunction = @(p) nnCostFunction(p, ...  
                                      layer_size_input, ...
                                      layer_size_hidden1,...
                                      labels_number, ...
                                      X, y, lambda);

% send a message to the user
fprintf('\n Training the neural net... \n');

% show time! running the optimizing alg
[nn_params, cost] = fmincg(costFunction, initial_nn_params,...
                            options);
                        
% recovering the trained theta matrices
Theta1 = reshape(nn_params(1:layer_size_hidden1 * (layer_size_input+1)),...
                 layer_size_hidden1, ...
                 (layer_size_input + 1)  );
             
Theta2 = reshape( nn_params( (1+(layer_size_hidden1*(layer_size_input+1))):end ),...
                   labels_number,...
                   (layer_size_hidden1 + 1));

%% post processing

% visualize weights
fprintf('\n Computed parameters: \n');
Theta1
Theta2

figure;
subplot(2,1,1);
imagesc(Theta1);
title('Theta 1');

subplot(2,1,2);
imagesc(Theta2);
title('Theta 2');

% predict to find accuracy
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining set accuracy: %f \n', mean(double(pred.' == y)) * 100);
















