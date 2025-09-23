function [center_site,cell_A_opt, cell_Y_hat, rmse_train_nocollab, mean_rmse_train_nocollab] = ...
    nocollab_model_train(splitdata,P,n_D,n_cluster,m)

n_split_data = zeros(P,1);
for i_site = 1:P
    n_split_data(i_site) = size(splitdata{i_site},1);
end

%% 每端的聚类中心和划分矩阵
tic;  %开始计时
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
    X_aug(:, 3) = 0; % 将第二列数据（对应 x2）全部置为0

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
        Y_hat(i) = sum(tmp_U(:, i) .* y_local') / sum(tmp_U(:, i));
    end
    cell_Y_hat{i_site} = Y_hat;
    rmse_train_nocollab(i_site) = sqrt(mean((tmp_y - Y_hat).^2));
end

time_nocollab = toc; %结束计时

mean_rmse_train_nocollab = mean(rmse_train_nocollab);  %每个客户端的RMSE求均值

% disp(['每端独立运行的训练集 RMSE 值为: ', num2str(mean_rmse_train_nocollab)]);
% disp(['每端独立运行的算法运行时间为: ', num2str(time_nocollab), ' 秒']);