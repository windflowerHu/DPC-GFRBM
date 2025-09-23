function [IA, IS] = evaluate_intervals(final_y_intervals, y_list)
    % 输入参数：
    % final_y_intervals: n x 2 矩阵，每行表示一个区间 [y_lower, y_upper]
    % y_list: n x 1 向量，表示精确 y 值
    % alpha: 惩罚参数 (通常为 0.05 或 0.1)
    %
    % 输出参数：
    % IA: 区间精度
    % IS: 区间得分

    alpha = 0.1;

    % 确保输入矩阵和向量尺寸匹配
    assert(size(final_y_intervals, 1) == length(y_list), '尺寸不匹配');
    
    % 提取区间上下界
    y_lower = final_y_intervals(:, 1);
    y_upper = final_y_intervals(:, 2);
    
    % 计算区间中心
    y_center = (y_lower + y_upper) / 2;
    
    % 计算区间宽度
    interval_width = abs(y_upper - y_lower);
    
    % 区间精度 (IA)
    IA = mean(abs(y_list - y_center) ./ (interval_width + eps)); % eps 防止除以零

    % 区间得分 (IS)
    penalties = max(0, y_lower - y_list) + max(0, y_list - y_upper); % 惩罚项
    IS = mean(interval_width + 2/alpha * penalties);
end