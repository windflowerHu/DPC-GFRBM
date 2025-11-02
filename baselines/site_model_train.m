function [cell_A_opt, cell_Y_hat, rmse,mean_rmse] = site_model_train(splitdata,P,site_U,n_cluster,n_D)
%% Training model at each site
cell_Y_hat = cell(P,1);
rmse = zeros(1,P);
cell_A_opt = cell(P,1);
parfor i_site = 1:P
    tmp_data = splitdata{i_site};
    tmp_y = tmp_data(:,end);
    tmp_data = tmp_data(:,1:end-1);
    tmp_U = site_U{i_site};
    tmp_U = tmp_U';
    tmp_n_data = size(tmp_data,1);
    X_aug = [ones(tmp_n_data,1),tmp_data];

    A = zeros(n_cluster,n_D);
    
    % Define objective function to minimize RMSE
    objectiveFunction = @(A_flat) compute_rmse(A_flat, tmp_U, X_aug, tmp_y, n_cluster, n_D-1);
    
    % Flatten matrix A for optimization
    A_flat_init = reshape(A, [], 1);
    
    % Use fminunc to minimize the RMSE
    options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'off');
    A_flat_opt = fminunc(objectiveFunction, A_flat_init, options);
    

    %% RMSE
    % Reshape optimized A_flat back to matrix form
    A_opt = reshape(A_flat_opt, n_cluster, n_D);
    cell_A_opt{i_site} = A_opt;
    
    % Fuzzy inference: compute predicted output for each data point
    Y_hat = zeros(tmp_n_data, 1);
    for i = 1:tmp_n_data
        % Compute the local predictions for each cluster
        y_local = X_aug(i, :) * A_opt'; % 1-by-c vector of local predictions
        % Weighted average of local predictions
        Y_hat(i) = sum(tmp_U(:, i) .* y_local') / sum(tmp_U(:, i));
    end
    cell_Y_hat{i_site} = Y_hat;
    rmse(i_site) = sqrt(mean((tmp_y - Y_hat).^2));
end
mean_rmse = mean(rmse);