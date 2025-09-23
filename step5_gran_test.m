function [U_global_R1_test]= step5_gran_test(data_test,m,n_cluster,n_D,center_global_R1)

%% 代入测试数据获得全局隶属度矩阵
% 求区间值原型的平均数，得到全局精确值原型
center_global_avg = zeros(size(center_global_R1, 1), size(center_global_R1, 2) / 2);
for i = 1:size(center_global_R1, 1)
    for j = 1:(size(center_global_R1, 2) / 2)
        min_val = center_global_R1(i, 2*j-1);
        max_val = center_global_R1(i, 2*j);
        center_global_avg(i, j) = (min_val + max_val) / 2;
    end
end
% disp(['center_global_avg: ', num2str(size(center_global_avg))]);

% 设置 FCM 参数
m = 2; % 模糊系数
max_iter = 200; % 最大迭代次数
tol = 1e-5; % 收敛容限

tmp_data = data_test(:, 1:end-1); % 去掉最后一列
n_samples = size(tmp_data, 1);
% 初始化隶属度矩阵
U = rand(n_samples, n_cluster);
U = U ./ sum(U, 2);

% 迭代更新隶属度矩阵
for iter = 1:max_iter
    % 计算数据点与聚类中心之间的距离
    dist = pdist2(tmp_data, center_global_avg, 'euclidean');

    % 更新隶属度矩阵
    U_new = zeros(size(U));
    for i = 1:n_samples
        for j = 1:n_cluster
            % 计算隶属度
            sum_inv_dist = sum(1 ./ (dist(i,:) .^ (2 / (m - 1))));
            U_new(i, j) = 1 / sum((1 ./ (dist(i, :) .^ (2 / (m - 1))))) .* (1 ./ (dist(i, j) .^ (2 / (m - 1))));
        end
    end

    % 更新隶属度矩阵
    U = U_new;

    % 检查收敛条件
    if max(abs(U - U_new), [], 'all') < tol
        break;
    end
end

% 存储隶属度矩阵
U_global_R1_test = U;

