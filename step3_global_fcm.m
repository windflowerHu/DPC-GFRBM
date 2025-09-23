function [center_global_R1, U_global_R1,U_global_site_R1,client_centers]= step3_global_fcm(splitdata,P,m,n_cluster,n_D,local_interval_cntrs)

%% 获得本地区间值原型
% 初始化一个cell数组，用于存储每个客户端的聚类中心数量
cluster_counts = zeros(P, 1);
for i = 1:P
    cluster_counts(i) = size(local_interval_cntrs{i}, 1); % 获取每个客户端的聚类数量
end

% 计算总聚类中心数量
total_clusters = sum(cluster_counts);

% 初始化一个二维数组用于存储每个原型的特征的最小值和最大值
all_centers = zeros(total_clusters, (n_D - 1) * 2);

% 初始化一个cell数组，用于按客户端存储聚类中心
client_centers = cell(P, 1);

% % 提取最小值和最大值并重组数据
% index = 1;
% for i = 1:P
%     for j = 1:cluster_counts(i) % 使用动态获取的聚类数量
%         % 获取当前聚类中心的特征三元组
%         center = local_interval_cntrs{i}{j}; % 获取cell数组
% 
%         % 提取每个特征的最小值和最大值并组成二元组
%         feature_vector = zeros(1, (n_D - 1) * 2);
%         for k = 1:n_D - 1
%             % 提取当前特征的下界和上界
%             min_val = center{k}{1}; % 下界值
%             max_val = center{k}{2}; % 上界值
% 
%             feature_vector(2*k - 1) = min_val; % 将下界值放入特征向量
%             feature_vector(2*k) = max_val; % 将上界值放入特征向量
%         end
% 
%         % 将二元组按特征顺序组成每行数据
%         all_centers(index, :) = feature_vector;
% 
%         index = index + 1;
%     end
% end


% 提取最小值和最大值并重组数据
index = 1;
for i = 1:P
    % 根据当前客户端的聚类中心数量初始化临时矩阵
    client_data = zeros(cluster_counts(i), (n_D - 1) * 2);
    
    for j = 1:cluster_counts(i) % 使用动态获取的聚类数量
        % 获取当前聚类中心的特征三元组
        center = local_interval_cntrs{i}{j}; % 获取cell数组

        % 提取每个特征的最小值和最大值并组成二元组
        feature_vector = zeros(1, (n_D - 1) * 2);
        for k = 1:n_D - 1
            % 提取当前特征的下界和上界
            min_val = center{k}{1}; % 下界值
            max_val = center{k}{2}; % 上界值
            
            feature_vector(2*k - 1) = min_val; % 将下界值放入特征向量
            feature_vector(2*k) = max_val; % 将上界值放入特征向量
        end

        % 将二元组按特征顺序组成每行数据，并存入all_centers和client_data
        all_centers(index, :) = feature_vector;
        client_data(j, :) = feature_vector;

        index = index + 1;
    end
    
    % 将当前客户端的所有聚类中心存储到client_centers中
    client_centers{i} = client_data;
end



%% 二次聚类得到全局区间值原型
tmp_n_data = size(all_centers,1);
tmp_indx = randperm(tmp_n_data, n_cluster);
center_ini = all_centers(tmp_indx,:);
[center_global_R1] = FCM(all_centers, n_cluster, m, 200, 1e-5, center_ini);



%% 代入原始数据获得全局隶属度矩阵

% 设置 FCM 参数
m = 2; % 模糊系数
max_iter = 200; % 最大迭代次数
tol = 1e-5; % 收敛容限

% 初始化 U_global_R1
U_global_site_R1 = cell(P, 1);

for i_site = 1:P
    tmp_data = splitdata{i_site};
    tmp_data = tmp_data(:, 1:end-1); % 去掉最后一列
    n_samples = size(tmp_data, 1);

    % 初始化隶属度矩阵
    U = rand(n_samples, n_cluster);
    U = U ./ sum(U, 2);

    % % 方法一：求 Hausdorff 距离
    % dist_hausdorff = zeros(n_samples, n_cluster);
    % for i = 1:n_samples
    %     for j = 1:n_cluster
    %         min_vals = center_global_R1(j, 1:2:end); % 区间下限
    %         max_vals = center_global_R1(j, 2:2:end); % 区间上限
    %         min_dist = min(abs(tmp_data(i, :) - min_vals), abs(tmp_data(i, :) - max_vals));
    %         dist_hausdorff(i, j) = max(min_dist); % 取最大值
    %     end
    % end

    % % 方法二：区间平均距离
    % dist_avg = zeros(n_samples, n_cluster);
    % for j = 1:n_cluster
    %     center_avg = mean([center_global_R1(j, 1:2:end); center_global_R1(j, 2:2:end)], 1); % 区间平均值
    %     dist_avg(:, j) = vecnorm(tmp_data - center_avg, 2, 2); % 欧氏距离
    % end

    % % 方法三：加权边界距离
    % alpha = 0.5; % 权重参数
    % dist_weighted = zeros(n_samples, n_cluster);
    % for i = 1:n_samples
    %     for j = 1:n_cluster
    %         min_vals = center_global_R1(j, 1:2:end);
    %         max_vals = center_global_R1(j, 2:2:end);
    %         dist_weighted(i, j) = alpha * norm(tmp_data(i, :) - min_vals) + (1 - alpha) * norm(tmp_data(i, :) - max_vals);
    %     end
    % end

    % % 方法四：Mahalanobis 距离
    % dist_mahalanobis = zeros(n_samples, n_cluster);
    % for j = 1:n_cluster
    %     center_avg = mean([center_global_R1(j, 1:2:end); center_global_R1(j, 2:2:end)], 1); % 区间平均值
    %     dist_mahalanobis(:, j) = mahal(tmp_data, center_avg); % Mahalanobis 距离
    % end

    % 方法五：中点与边界联合距离
    dist_combined = zeros(n_samples, n_cluster);
    for i = 1:n_samples
        for j = 1:n_cluster
            center_avg = mean([center_global_R1(j, 1:2:end); center_global_R1(j, 2:2:end)], 1); % 区间中点
            min_vals = center_global_R1(j, 1:2:end);
            max_vals = center_global_R1(j, 2:2:end);
            dist_combined(i, j) = 0.5 * (norm(tmp_data(i, :) - center_avg) + min(norm(tmp_data(i, :) - min_vals), norm(tmp_data(i, :) - max_vals)));
        end
    end


    % 选择一种距离矩阵作为隶属度计算基础，例如 Hausdorff 距离
    dist = dist_combined; 

    % 迭代更新隶属度矩阵
    for iter = 1:max_iter
        % 更新隶属度矩阵
        U_new = zeros(size(U));

        % % 计算数据点与聚类中心之间的距离
        % dist = pdist2(tmp_data, center_global_avg, 'euclidean');

        for i = 1:n_samples
            for j = 1:n_cluster
                % 计算隶属度
                sum_inv_dist = sum(1 ./ (dist(i,:) .^ (2 / (m - 1))));
                U_new(i, j) = (1 / dist(i, j) .^ (2 / (m - 1))) / sum_inv_dist;
                % U_new(i, j) = 1 / sum((1 ./ (dist(i, :) .^ (2 / (m - 1))))) .* (1 ./ (dist(i, j) .^ (2 / (m - 1))));
            end
        end

        % 检查收敛条件
        if max(abs(U - U_new), [], 'all') < tol
            break;
        end
        % 更新隶属度矩阵
        U = U_new;

    end

    % 存储隶属度矩阵
    U_global_site_R1{i_site} = U;
    % disp(['U_global_site_R1{i_site}: ', num2str(size(U_global_site_R1{i_site}))]);
end

% 将每个 U_global_site_R1 按列合并成一个大矩阵 U_global_R1
U_global_R1 = vertcat(U_global_site_R1{:});




% %% 代入原始数据获得全局隶属度矩阵
% 
% % 方法一：求区间值原型的平均数，得到全局精确值原型
% center_global_avg = zeros(size(center_global_R1, 1), size(center_global_R1, 2) / 2);
% for i = 1:size(center_global_R1, 1)
%     for j = 1:(size(center_global_R1, 2) / 2)
%         min_val = center_global_R1(i, 2*j-1);
%         max_val = center_global_R1(i, 2*j);
%         center_global_avg(i, j) = (min_val + max_val) / 2;
%     end
% end
% % disp(['center_global_avg: ', num2str(size(center_global_avg))]);
% 
% % 设置 FCM 参数
% m = 2; % 模糊系数
% max_iter = 200; % 最大迭代次数
% tol = 1e-5; % 收敛容限
% 
% % 初始化 U_global_R1
% U_global_site_R1 = cell(P, 1);
% 
% for i_site = 1:P
%     tmp_data = splitdata{i_site};
%     tmp_data = tmp_data(:, 1:end-1); % 去掉最后一列
%     n_samples = size(tmp_data, 1);
% 
%     % 初始化隶属度矩阵
%     U = rand(n_samples, n_cluster);
%     U = U ./ sum(U, 2);
% 
%     % 方法一：求 Hausdorff 距离
%     dist_hausdorff = zeros(n_samples, n_cluster);
%     for i = 1:n_samples
%         for j = 1:n_cluster
%             min_vals = center_global_R1(j, 1:2:end); % 区间下限
%             max_vals = center_global_R1(j, 2:2:end); % 区间上限
%             min_dist = min(abs(tmp_data(i, :) - min_vals), abs(tmp_data(i, :) - max_vals));
%             dist_hausdorff(i, j) = max(min_dist); % 取最大值
%         end
%     end
% 
% 
% 
%     % 迭代更新隶属度矩阵
%     for iter = 1:max_iter
%         % 计算数据点与聚类中心之间的距离
%         dist = pdist2(tmp_data, center_global_avg, 'euclidean');
% 
%         % 更新隶属度矩阵
%         U_new = zeros(size(U));
%         for i = 1:n_samples
%             for j = 1:n_cluster
%                 % 计算隶属度
%                 sum_inv_dist = sum(1 ./ (dist(i,:) .^ (2 / (m - 1))));
%                 U_new(i, j) = 1 / sum((1 ./ (dist(i, :) .^ (2 / (m - 1))))) .* (1 ./ (dist(i, j) .^ (2 / (m - 1))));
%             end
%         end
% 
%         % 更新隶属度矩阵
%         U = U_new;
% 
%         % 检查收敛条件
%         if max(abs(U - U_new), [], 'all') < tol
%             break;
%         end
% 
% 
%     end
% 
%     % 存储隶属度矩阵
%     U_global_site_R1{i_site} = U;
%     % disp(['U_global_site_R1{i_site}: ', num2str(size(U_global_site_R1{i_site}))]);
% end
% 
% % 将每个 U_global_site_R1 按列合并成一个大矩阵 U_global_R1
% U_global_R1 = vertcat(U_global_site_R1{:});
