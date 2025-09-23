function [client_prototypes, client_membership, granularity_results] = dbscan_model2(splitdata, P, n_cluster)
    % DBSCAN_FCM_PROTOTYPES: 通过 DBSCAN 划分数据集后，使用 FCM 对每个子集聚类，
    % 并随机从每个子集选择初始聚类中心。
    % 
    % 输入参数：
    % - splitdata: 每个客户端的数据（元胞数组）
    % - P: 客户端数量
    % - n_cluster: 每个子数据集进行 FCM 时的目标聚类数
    % 
    % 输出参数：
    % - client_prototypes: 各客户端的聚类中心（元胞数组，大小为 P×DBSCAN聚类中心数×FCM聚类中心数）
    % - client_membership: 各客户端原始数据对其聚类中心的隶属度矩阵（元胞数组，大小为 P×DBSCAN聚类中心数×FCM聚类中心数）
    % - granularity_results: 各客户端聚类中心对应的 y 值区间（大小为 P×DBSCAN聚类中心数×FCM聚类中心数）

    eps = 1; 
    minPts = 2; 
    fuzziness = 2; % FCM 的模糊系数
    max_iter = 100; % FCM 的最大迭代次数
    tol = 1e-5; % FCM 的收敛阈值

    % 初始化结果
    client_prototypes = cell(P, 1);  % 存储各客户端的聚类中心
    client_membership = cell(P, 1); % 存储各客户端原始数据对其聚类中心的隶属度矩阵
    granularity_results = cell(P, 1); % 各客户端聚类中心对应的 y 值区间

    % 遍历每个客户端的数据
    for i_site = 1:P
        tmp_data = splitdata{i_site};
        tmp_data_x = tmp_data(:, 1:end-1); % 去掉最后一列（收益列）
        tmp_y = tmp_data(:, end);          % 提取收益列

        % 使用 DBSCAN 聚类将原始数据划分为多个小数据集
        labels = dbscan(tmp_data_x, eps, minPts);

        % 获取所有有效的聚类标签
        unique_labels = unique(labels(labels >= 0)); % 排除噪声点 (label = -1)
        num_dbscan_clusters = length(unique_labels);
        
        % % 输出 DBSCAN 聚类数量
        % disp(['客户端 ', num2str(i_site), ' DBSCAN 聚类数量: ', num2str(num_dbscan_clusters)]);

        % 初始化该客户端的结果
        client_prototypes{i_site} = []; % 用于存储合并后的聚类中心
        client_membership{i_site} = []; % 用于存储合并后的隶属度矩阵
        granularity_results{i_site} = []; % 用于存储合并后的粒度结果

        % 遍历每个DBSCAN聚类子集
        for i_cluster = 1:num_dbscan_clusters
            % 提取属于当前子集的数据
            sub_data_idx = labels == unique_labels(i_cluster);
            sub_data_x = tmp_data_x(sub_data_idx, :);
            sub_data_y = tmp_y(sub_data_idx);

            % 如果子数据集为空，跳过
            if isempty(sub_data_x)
                continue;
            end

            % 随机选择子集中的数据点作为初始聚类中心
            num_points = size(sub_data_x, 1);
            if num_points < n_cluster
                % 如果数据点不足以初始化所有中心，随机重复数据点
                init_centers = sub_data_x(randi(num_points, n_cluster, 1), :);
            else
                % 随机选择不重复的初始中心
                init_centers = sub_data_x(randperm(num_points, n_cluster), :);
            end

            % 使用 FCM 对子数据集进行聚类
            [fcm_centers, fcm_membership] = FCM(sub_data_x, n_cluster, fuzziness, max_iter, tol, init_centers);

            % 将该 DBSCAN 聚类中心对应的 FCM 聚类中心合并
            client_prototypes{i_site} = [client_prototypes{i_site}; fcm_centers];

            % % 输出 FCM 聚类数量
            % disp(['客户端 ', num2str(i_site), ' 的 FCM 聚类数量（子集 ', num2str(i_cluster), '）: ', num2str(n_cluster)]);

            % 计算该子数据集中的隶属度矩阵
            dist_matrix = pdist2(tmp_data_x, fcm_centers); % 原始数据到新聚类中心的距离
            sub_membership = 1 ./ (dist_matrix + eps); % 转换为隶属度 (防止除零)
            sub_membership = sub_membership ./ sum(sub_membership, 2); % 归一化

            % 合并该 DBSCAN 聚类中心对应的隶属度矩阵
            client_membership{i_site} = [client_membership{i_site}, sub_membership];

            % 存储该 DBSCAN 聚类中心的粒度结果
            for j = 1:n_cluster
                granularity_result = step4_gran_y([tmp_data_x, tmp_y], sub_membership(:, j));

                % 将每个 FCM 聚类中心的粒度结果合并到 granularity_results 中
                granularity_results{i_site} = [granularity_results{i_site}; granularity_result];
            end
        end
    end

    % % 输出结果维度
    % disp(['client_prototypes{1} ', num2str(i_site), ' : ', num2str(size(client_prototypes{1}))]);
    % disp(['client_membership{1} ', num2str(i_site), ' : ', num2str(size(client_membership{1}))]);
end
