function [client_prototypes, client_membership, granularity_results] = dbscan_model3(splitdata, P, n_cluster)
    % DBSCAN_FCM_PROTOTYPES: 用于通过 FCM 聚类各个客户端的原型
    % 并存储各客户端原始数据对于其原型的隶属度矩阵。
    % 输入参数： 
    % - splitdata: 每个客户端的数据（元胞数组） 
    % - P: 客户端数量 
    % - n_cluster: 聚类中心数量

    fuzziness = 2;       % FCM 模糊系数
    max_iter = 100;      % 最大迭代次数
    tol = 1e-5;          % 收敛阈值

    % 初始化结果
    client_prototypes = cell(1, P);  % 存储各客户端的原型
    client_membership = cell(1, P);  % 存储各客户端原始数据对其原型的隶属度矩阵
    granularity_results = cell(1, P); % 各客户端原型对应的 y 值区间

    % FCM 聚类
    for i_site = 1:P
        tmp_data = splitdata{i_site};
        tmp_data_x = tmp_data(:, 1:end-1); % 去掉最后一列（收益列）
        tmp_y = tmp_data(:, end);           % 提取收益列

        % 初始化聚类中心
        initial_centers = tmp_data_x(1:n_cluster, :);  % 随机选择初始聚类中心

        % 使用 FCM 聚类
        [centers, U] = FCM(tmp_data_x, n_cluster, fuzziness, max_iter, tol, initial_centers);

        % 存储该客户端的原型
        client_prototypes{i_site} = centers;

        % 存储隶属度矩阵
        client_membership{i_site} = U;

        % 输出聚类中心和隶属度矩阵的维度
        disp(['客户端 ', num2str(i_site), ' 的聚类中心维度: ', num2str(size(centers))]);
        disp(['客户端 ', num2str(i_site), ' 的隶属度矩阵维度: ', num2str(size(U))]);

        % 调用合理粒度原则函数计算 y 值区间
        granularity_results{i_site} = step4_gran_y([tmp_data_x, tmp_y], U);

    end
end
