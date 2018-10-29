function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% Plotting approved
indi_approved = find(y == 1);
scatter(X(indi_approved,1),X(indi_approved,2),'b+','LineWidth',2);

% Plottin disapproved
indi_disapproved = find(y == 0);
scatter(X(indi_disapproved,1),X(indi_disapproved,2),'ro','LineWidth',2);

% Figure implementing.
title('Last years student approval');
xlabel('Exam 1 score'); ylabel('Exam 2 score');
grid on;
legend('Approved','Disapproved');



% =========================================================================



hold off;

end
