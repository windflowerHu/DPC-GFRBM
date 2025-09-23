function [results, final_y_intervals] = step4_new_gran_y(data_train, U_global_R1)
    % 提取 y 值并定义初始参数
    y_list = data_train(:, end); % 提取收益 y 值
    numClusters = size(U_global_R1, 2); % 聚类中心数量
    step_size = 0.1; % 步长
    results = zeros(numClusters, 8); % 存储 [上界目标函数最大值, 上界覆盖率, 上界特异性, 下界目标函数最大值, 下界覆盖率, 下界特异性, 上界, 下界]
    range_y = max(y_list) - min(y_list) + 1e-12; % 数据的整体范围，加上小值防止除零
    
    % 初始化最终的 y 区间列表
    final_y_intervals = zeros(size(y_list, 1), 2); % 每条数据的 y 区间 [下界，上界]
    
    % 找到每行最大值的索引
    [~, maxIndices] = max(U_global_R1, [], 2); 

    % 遍历每个聚类中心以确定最佳区间
    for clusterIdx = 1:numClusters
        % 获取属于当前聚类中心的 y 值
        belongingIndices = find(maxIndices == clusterIdx);
        if isempty(belongingIndices)
            y_values = median(y_list); % 若无数据，使用全体数据中位数
        else
            y_values = y_list(belongingIndices);
        end

        % 计算最大和最小值
        max_y = max(y_values);
        min_y = min(y_values);
        
        % 优化上界
        upper_bound_y = min_y; % 初始上界为最小值
        final_upper_bound = min_y;
        max_target_upper = 0;
        while upper_bound_y <= max_y
            % 计算覆盖度
            coverage_indices = (y_list >= min_y) & (y_list <= upper_bound_y);
            cov = sum(U_global_R1(coverage_indices, clusterIdx)) / sum(U_global_R1(:, clusterIdx));
            
            % 计算特异性
            sp = 1 - abs(upper_bound_y - min_y) / range_y;

            % 计算目标函数
            target = cov * sp;
            
            % 更新最佳上界
            if target > max_target_upper
                max_target_upper = target;
                final_upper_bound = upper_bound_y;
            end
            
            upper_bound_y = upper_bound_y + step_size;
        end
        
        % 优化下界
        lower_bound_y = min_y;
        final_lower_bound = min_y;
        max_target_lower = 0;
        while lower_bound_y <= max_y
            % 计算覆盖度
            coverage_indices = (y_list >= lower_bound_y) & (y_list <= max_y);
            cov = sum(U_global_R1(coverage_indices, clusterIdx)) / sum(U_global_R1(:, clusterIdx));
            
            % 计算特异性
            sp = 1 - abs(max_y - lower_bound_y) / range_y;

            % 计算目标函数
            target = cov * sp;
            
            % 更新最佳下界
            if target > max_target_lower
                max_target_lower = target;
                final_lower_bound = lower_bound_y;
            end
            
            lower_bound_y = lower_bound_y + step_size;
        end
        
        % 存储结果
        results(clusterIdx, :) = [max_target_upper, cov, sp, ...
                                  max_target_lower, cov, sp, ...
                                  final_upper_bound, final_lower_bound];
    end
    
    % 使用每个聚类中心的区间值计算每条数据的 y 区间
    for dataIdx = 1:size(y_list, 1)
        y_interval_lower = 0;
        y_interval_upper = 0;
        for clusterIdx = 1:numClusters
            membership = U_global_R1(dataIdx, clusterIdx);
            y_interval_lower = y_interval_lower + membership * results(clusterIdx, 8);
            y_interval_upper = y_interval_upper + membership * results(clusterIdx, 7);
        end
        final_y_intervals(dataIdx, :) = [y_interval_lower, y_interval_upper];
    end
end
