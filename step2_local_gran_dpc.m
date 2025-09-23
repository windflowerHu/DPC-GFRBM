function [local_interval_cntrs] = step2_local_gran_dpc(splitdata, P, center_site_local_dpc_R1, U_site_local_dpc_R1, n_D)
    % 初始化一个cell数组用于存储每个客户端的聚类中心和每个原型的特征三元组
    local_interval_cntrs = cell(P, 1);  % 初始化为P行，列数稍后根据聚类中心数量动态调整

    % alpha = 1; % 特异性参数

    % 计算区间值聚类中心
    for i_site = 1:P
        Z = splitdata{i_site};  % 从splitdata获取数据
        center_site = center_site_local_dpc_R1{i_site}; % 获取当前客户端的聚类中心
        U_site = U_site_local_dpc_R1{i_site};  % 获取当前客户端的隶属度矩阵
        Z = Z(:, 1:end-1);  % 去除最后一列（y）

        % 获取当前客户端的聚类中心数量
        n_cluster = size(center_site, 1);
        
        % 为每个客户端初始化一个cell数组用于存储聚类中心的区间
        local_interval_cntrs{i_site} = cell(n_cluster, 1); % 动态调整为每个客户端的聚类中心数量

        % 初始化存储上下界的向量
        lower_bounds = min(Z, [], 1); % 每个特征的下界
        upper_bounds = max(Z, [], 1); % 每个特征的上界

        % 遍历每个聚类中心
        for c = 1:n_cluster
            v = center_site(c, :); % 第 c 个原型（包含了 n_D 个特征）
            local_interval_cntrs{i_site}{c} = cell(1, n_D - 1); % 初始化每个原型的特征三元组

            % 遍历每个特征
            for i = 1:n_D - 1
                % 生成a和b的候选值
                a_candi = lower_bounds(i):0.05:v(i);  % 下界范围
                b_candi = v(i):0.05:upper_bounds(i);  % 上界范围
                range_a = abs(v(i) - lower_bounds(i)); 
                range_b = abs(upper_bounds(i) - v(i));
                
                % 计算a的隶属度结果
                result_a = zeros(1, numel(a_candi)); 
                for a_idx = 1:numel(a_candi)
                    cov_a = sum(U_site(c, Z(:, i) >= a_candi(a_idx) & Z(:, i) <= v(i)));
                    sp_a = 1 - abs(a_candi(a_idx) - v(i)) / range_a;
                    result_a(a_idx) = cov_a * sp_a;
                end

                % 计算b的隶属度结果
                result_b = zeros(1, numel(b_candi));
                for b_idx = 1:numel(b_candi)
                    cov_b = sum(U_site(c, Z(:, i) >= v(i) & Z(:, i) <= b_candi(b_idx)));
                    sp_b = 1 - abs(b_candi(b_idx) - v(i)) / range_b;
                    result_b(b_idx) = cov_b * sp_b;
                end
                
                % 找到乘积最大的best_a和best_b
                [~, idx_a_max] = max(result_a);
                best_a = a_candi(idx_a_max);

                [~, idx_b_max] = max(result_b);
                best_b = b_candi(idx_b_max);

                % 存储特征的下界、上界和精确值到cell数组中
                local_interval_cntrs{i_site}{c}{i} = {best_a, best_b, v(i)};
            end
        end
    end
end
