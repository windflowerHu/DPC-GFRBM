function [center_site,cell_A_opt, cell_Y_hat, rmse_train_nocollab, mean_rmse_train_nocollab] = ...
    nocollab_model_train(splitdata,P,n_D,n_cluster,m)

n_split_data = zeros(P,1);
for i_site = 1:P
    n_split_data(i_site) = size(splitdata{i_site},1);
end

%% Clustering centers and partition matrix for each site
center_site = cell(P,1);
U_site = cell(P,1);
for i_site = 1:P
    tmp_data = splitdata{i_site};
    tmp_n_data = size(tmp_data,1);
    tmp_indx = randperm(tmp_n_data, n_cluster);
    center_ini = tmp_data(tmp_indx,:);
    [center_site{i_site},U_site{i_site}] = FCM(tmp_data, n_cluster, m, 200, 1e-5, center_ini);
    U_site{i_site} = U_site{i_site}';
end


cell_A = cell(P,1);
cell_Y_hat = cell(P,1);
rmse_train_nocollab = zeros(1,P);
cell_A_opt = cell(P,1);
parfor i_site = 1:P
    tmp_data = splitdata{i_site};
    tmp_y = tmp_data(:,end);
    tmp_data = tmp_data(:,1:end-1);
    tmp_U = U_site{i_site};
    tmp_n_data = n_split_data(i_site);
    X_aug = [ones(tmp_n_data,1),tmp_data];

    A = zeros(n_cluster,n_D);

    % Define objective function to minimize RMSE
    objectiveFunction = @(A_flat) compute_rmse(A_flat, tmp_U, X_aug, tmp_y, n_cluster, n_D-1);

    % Flatten matrix A for optimization
    A_flat_init = reshape(A, [], 1);

    % Use fminunc to minimize the RMSE
    options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'off');
    A_flat_opt = fminunc(objectiveFunction, A_flat_init, options);

    % Reshape optimized A_flat back to matrix form
    cell_A{i_site} = reshape(A_flat_opt, n_cluster, n_D);

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
        u_sum = sum(tmp_U(:, i));
        if u_sum > 1e-10
            Y_hat(i) = sum(tmp_U(:, i) .* y_local') / u_sum;
        else
            % If membership sum is too small, use simple average
            Y_hat(i) = mean(y_local);
        end
    end
    cell_Y_hat{i_site} = Y_hat;
    rmse_train_nocollab(i_site) = sqrt(mean((tmp_y - Y_hat).^2));
end

mean_rmse_train_nocollab = mean(rmse_train_nocollab);
