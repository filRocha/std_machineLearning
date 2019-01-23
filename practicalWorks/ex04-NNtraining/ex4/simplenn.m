clear;
close all;
clc;

% input data
X = [0, 0, 1; 0, 1, 1; 1, 0, 1; 1, 1, 1];

% output data
y = [0; 1; 1; 0];

% sinapses
syn0 = 2 * rand(3,4) - 1;
syn1 = 2 * rand(4,1) - 1;

% training loop
for j=1:60000
    
    % forward propagation
    l0 = X;
    
    z1 = l0 * syn0;
    l1 = minhafunc(z1, false);
    
    z2 = l1 * syn1;
    l2 = minhafunc(z2, false);
    
    % backpropagation
    l2_error = y - l2;
    l2_delta = l2_error * minhafunc(l2, true);
    
    l1_error = l2_delta * syn1;
    l1_delta = l1_error * minhafunc(l1, true).';
    
    % displays the error rate
    if rem(j,1000) == 0
       disp(abs(mean(l2_error))); 
    end
    
    % update weights
    syn1 = syn1 + l1.' * l2_delta;
    syn0 = syn0 + l0.' * l1_delta;
    
end