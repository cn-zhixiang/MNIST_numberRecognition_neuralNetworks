clear;
clc;
close all;

start_time = time;

[X y] = loadData(1);
[X_t y_t] = loadData(0);

[m n] = size(X);
n_2 = 300; 

theta = fmin_nn (n_2, X, y, 0.1);

pred = predict_nn(theta, n_2, X);
fprintf('Training set accuracy %.2f...\n', mean(double(pred==y))*100);

pred_t = predict_nn(theta, n_2, X_t);
fprintf('Test set accuracy %.2f...\n', mean(double(pred_t==y_t))*100);

end_time = time;
diff = end_time-start_time;
fprintf('Training took %.2f seconds\n', diff);

fprintf('Program finished...\n');