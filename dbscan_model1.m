function [client_prototypes, client_membership, granularity_results] = dbscan_model1(splitdata, P)
    % DBSCAN_FCM_PROTOTYPES: 用于通过 DBSCAN 聚类各个客户端的原型，
    % 并存储各客户端原始数据对于其原型的隶属度矩阵。
    %
    % 输入参数：
    % - splitdata: 每个客户端的数据（元胞数组）
    % - P: 客户端数量
    % - eps: DBSCAN 的邻域半径
    % - minPts: DBSCAN 的最小点数
    %
    % 输出参数：
    % - client_prototypes: 各客户端的原型（元胞数组，大小为 P）
    % - client_membership: 各客户端原始数据对其原型的隶属度矩阵（元胞数组，大小为 P）

    eps = 1; 
    minPts = 2; 


    % 初始化结果
    client_prototypes = cell(1, P);  % 存储各客户端的原型
    client_membership = cell(1, P); % 存储各客户端原始数据对其原型的隶属度矩阵
    granularity_results = cell(1, P);       % 各客户端原型对应的 y 值区间

    % DBSCAN 找出每个客户端的原型
    for i_site = 1:P
        tmp_data = splitdata{i_site};
        tmp_data_x = tmp_data(:, 1:end-1); % 去掉最后一列（收益列）
        tmp_y = tmp_data(:, end);           % 提取收益列

        % 使用 DBSCAN 聚类
        labels = dbscan(tmp_data_x, eps, minPts);

        % 筛选非噪声点并计算聚类中心
        unique_labels = unique(labels(labels >= 0)); % 排除噪声点 (label = -1)
        tmp_centers = zeros(length(unique_labels), size(tmp_data_x, 2));
        for i = 1:length(unique_labels)
            tmp_centers(i, :) = mean(tmp_data_x(labels == unique_labels(i), :), 1);
        end

        % 存储该客户端的原型
        client_prototypes{i_site} = tmp_centers;

        % 计算隶属度矩阵
        if ~isempty(tmp_centers)
            % 计算每个原始数据到原型的距离
            dist_matrix = pdist2(tmp_data_x, tmp_centers); % 原始数据到聚类中心的距离
            % 转换为隶属度 (基于距离的反比)
            membership = 1 ./ (dist_matrix + eps); % 防止除零，加上一个小的 eps
            % 归一化隶属度
            membership = membership ./ sum(membership, 2);
            client_membership{i_site} = membership;

            % 调用合理粒度原则函数计算 y 值区间
            granularity_results{i_site} = step4_gran_y([tmp_data_x, tmp_y], membership);
        else
            client_membership{i_site} = []; % 如果没有原型，则隶属度为空
            granularity_results{i_site} = [];   % 如果没有原型，则粒度结果为空
        end

        % 输出当前客户端的信息
        disp(['客户端 ', num2str(i_site), ' 的原型数量为: ', num2str(size(tmp_centers, 1))]);
    end
end
