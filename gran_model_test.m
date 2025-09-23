function [rmse_test,mean_rmse_test,cell_Y_hat_test] = gran_model_test(test_data, P, center_site, cell_A_opt, m, n_cluster)
ground_truth_test_y = test_data(:,end);
% test_data = test_data(:,1:end-1);
[n_data_test,n_D] = size(test_data);
X_aug_test = [ones(n_data_test,1), test_data(:,1:end-1)];

cell_Y_hat_test = zeros(n_data_test,P);
rmse_test = zeros(1,P);
parfor i_site = 1:P
    tmp_center_site = center_site{i_site};
    dist_center_data_test = pdist2(tmp_center_site,test_data);
    U_test = 1.0./(dist_center_data_test./(ones(n_cluster,1)*sum(dist_center_data_test))).^(2.0/(m-1));
    % Fuzzy inference: compute predicted output for each data point
    Y_hat = zeros(n_data_test, 1);
    A_opt = cell_A_opt{i_site};
    for i = 1:n_data_test
        % Compute the local predictions for each cluster
        y_local = X_aug_test(i, :) * A_opt'; % 1-by-c vector of local predictions
        % Weighted average of local predictions
        Y_hat(i) = sum(U_test(:, i) .* y_local') / sum(U_test(:, i));
    end
    cell_Y_hat_test(:,i_site) = Y_hat;

    rmse_test(i_site) = sqrt(mean((ground_truth_test_y - Y_hat).^2));
end
mean_rmse_test = mean(rmse_test);

