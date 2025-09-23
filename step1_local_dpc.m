function [center_site_local_dpc_R1, U_site_local_dpc_R1] = step1_local_dpc(splitdata, P)
    % 初始化输出变量
    center_site_local_dpc_R1 = cell(P, 1);  % 存储每个客户端的聚类中心
    U_site_local_dpc_R1 = cell(P, 1);       % 存储每个客户端的隶属度矩阵

    % 对每个客户端进行DPC聚类
    for i_site = 1:P
        data = splitdata{i_site}; % 当前客户端的数据
        data = data(:, 1:end-1);
        N = size(data, 1);

        % 计算距离矩阵
        distance_matrix = pdist2(data, data);

        % 设置截断距离dc
        % percent = 2; % 截断距离取距离矩阵中百分之2的距离
        sorted_distances = sort(distance_matrix(:));
        % dc = sorted_distances(ceil(percent/100 * length(sorted_distances)));
        dc = 0.02;

        % 计算每个点的局部密度 rho
        rho = sum(exp(-(distance_matrix / dc).^2), 2);

        % 计算每个点到最近高密度点的距离 delta
        delta = inf(N, 1);
        for i = 1:N
            higher_density_indices = find(rho > rho(i));
            if ~isempty(higher_density_indices)
                delta(i) = min(distance_matrix(i, higher_density_indices));
            else
                delta(i) = max(distance_matrix(i, :)); % 如果没有比它密度高的点，取最大距离
            end
        end

        % 选择聚类中心
        decision_values = rho .* delta;
        threshold = mean(decision_values) + std(decision_values);  % 聚类中心的动态选择
        % threshold = 5;  % 示例值，请根据实际情况调整
        center_indices = find(decision_values > threshold);
        cluster_centers = data(center_indices, :);
        num_centers = size(cluster_centers, 1);

        % 高斯核函数计算隶属度矩阵
        membership = zeros(N, num_centers);
        sigma = mean(sorted_distances);  % 核宽度选择为距离的均值
        for i = 1:N
            for j = 1:num_centers
                dist = norm(data(i, :) - cluster_centers(j, :));
                membership(i, j) = exp(-dist^2 / (2 * sigma^2));
            end
            % 归一化隶属度，使得每个点的隶属度和为1
            membership(i, :) = membership(i, :) / sum(membership(i, :));
        end

        % 存储当前客户端的聚类结果和隶属度矩阵
        center_site_local_dpc_R1{i_site} = cluster_centers;
        U_site_local_dpc_R1{i_site} = membership; % 直接将隶属度矩阵存入U_site_local_dpc_R1
        U_site_local_dpc_R1{i_site} = U_site_local_dpc_R1{i_site}';

        % 输出当前客户端的聚类中心数量
        % fprintf('客户端 %d 的聚类中心数量: %d\n', i_site, size(cluster_centers, 1));
    end
end
