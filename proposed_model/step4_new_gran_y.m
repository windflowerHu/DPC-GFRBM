function [results, final_y_intervals] = step4_new_gran_y(data_train, U_global_R1)
    y_list = data_train(:, end);
    numClusters = size(U_global_R1, 2);
    step_size = 0.1;
    results = zeros(numClusters, 8); % Store [upper bound max objective, upper bound coverage, upper bound specificity, lower bound max objective, lower bound coverage, lower bound specificity, upper bound, lower bound]
    range_y = max(y_list) - min(y_list) + 1e-12; % Overall data range, add small value to prevent division by zero
    final_y_intervals = zeros(size(y_list, 1), 2); % Y interval for each data point [lower bound, upper bound]
    
    % Find index of maximum value in each row
    [~, maxIndices] = max(U_global_R1, [], 2);
    
    % Iterate through each cluster center to determine optimal interval
    for clusterIdx = 1:numClusters
        % Get y values belonging to current cluster center
        belongingIndices = find(maxIndices == clusterIdx);
        if isempty(belongingIndices)
            y_values = median(y_list); % If no data, use median of all data
        else
            y_values = y_list(belongingIndices);
        end
        
        max_y = max(y_values);
        min_y = min(y_values);
        upper_bound_y = min_y;
        final_upper_bound = min_y;
        max_target_upper = 0;
        
        while upper_bound_y <= max_y
            % Calculate coverage
            coverage_indices = (y_list >= min_y) & (y_list <= upper_bound_y);
            cov = sum(U_global_R1(coverage_indices, clusterIdx)) / sum(U_global_R1(:, clusterIdx));
            
            % Calculate specificity
            sp = 1 - abs(upper_bound_y - min_y) / range_y;
            
            % Calculate objective function
            target = cov * sp;
            
            % Update optimal upper bound
            if target > max_target_upper
                max_target_upper = target;
                final_upper_bound = upper_bound_y;
            end
            upper_bound_y = upper_bound_y + step_size;
        end
        
        % Optimize lower bound
        lower_bound_y = min_y;
        final_lower_bound = min_y;
        max_target_lower = 0;
        
        while lower_bound_y <= max_y
            % Calculate coverage
            coverage_indices = (y_list >= lower_bound_y) & (y_list <= max_y);
            cov = sum(U_global_R1(coverage_indices, clusterIdx)) / sum(U_global_R1(:, clusterIdx));
            
            % Calculate specificity
            sp = 1 - abs(max_y - lower_bound_y) / range_y;
            
            % Calculate objective function
            target = cov * sp;
            
            % Update optimal lower bound
            if target > max_target_lower
                max_target_lower = target;
                final_lower_bound = lower_bound_y;
            end
            lower_bound_y = lower_bound_y + step_size;
        end
        
        % Store results
        results(clusterIdx, :) = [max_target_upper, cov, sp, ...
                                  max_target_lower, cov, sp, ...
                                  final_upper_bound, final_lower_bound];
    end
    
    % Calculate y interval for each data point using interval values from each cluster center
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