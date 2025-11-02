function [results] = comput_gran_y(data_train,U_global_R1)
%% Calculate granular prediction values based on justifiable granularity principle
y_list = data_train(:, end); % Extract last column as y values
numClusters = size(U_global_R1, 2); % Number of cluster centers
median_y_values = zeros(numClusters, 1); % Initialize array for median y values
max_y_values = zeros(numClusters, 1); % Initialize array for maximum y values
min_y_values = zeros(numClusters, 1); % Initialize array for minimum y values
step_size = 0.1; % Step size
results = zeros(numClusters, 8); % Store [upper bound max objective, upper bound coverage, upper bound specificity, lower bound max objective, lower bound coverage, lower bound specificity, upper bound, lower bound]
[~, maxIndices] = max(U_global_R1, [], 2); % Find index of maximum value in each row

for clusterIdx = 1:numClusters
    % Find data belonging to current cluster center
    belongingIndices = find(maxIndices == clusterIdx);
    
    % Get y values corresponding to these data points
    if isempty(belongingIndices)
        y_values = median(y_list); % Use median of all y values as y_values
    else
        y_values = y_list(belongingIndices);
    end
    
    % Calculate maximum, minimum, and median values
    median_y = median(y_values);
    max_y = max(y_values);
    min_y = min(y_values);
    range = abs(max_y - median_y); % Calculate range
    
    % Upper bound calculation
    max_cov_upper = 0;
    max_sp_upper = 0;
    max_target_upper = 0;
    upper_bound_y = median_y; % Initial upper bound is median
    final_upper_bound = median_y; % Store final upper bound
    
    while upper_bound_y <= max_y
        % Calculate coverage rate
        coverage_indices = (y_list >= median_y) & (y_list <= upper_bound_y);
        if any(coverage_indices)
            cov = sum(U_global_R1(coverage_indices, clusterIdx));
        else
            cov = 0;
        end
        sp = 1 - abs(upper_bound_y - median_y) / (range + 1e-12);
        
        % Calculate objective function
        target = cov * sp;
        
        % Update maximum values
        if target > max_target_upper
            max_target_upper = target;
            max_cov_upper = cov;
            max_sp_upper = sp;
            final_upper_bound = upper_bound_y; % Update final upper bound
        end
        upper_bound_y = upper_bound_y + step_size; % Update upper bound
    end
    
    % Lower bound calculation
    max_cov_lower = 0;
    max_sp_lower = 0;
    max_target_lower = 0;
    lower_bound_y = min_y; % Initial lower bound is minimum value
    final_lower_bound = min_y; % Store final lower bound
    
    while lower_bound_y <= median_y
        % Calculate coverage rate
        coverage_indices = (y_list >= lower_bound_y) & (y_list <= median_y);
        if any(coverage_indices)
            cov = sum(U_global_R1(coverage_indices, clusterIdx)); % Coverage rate
        else
            cov = 1e-6;
        end
        
        % Calculate specificity
        sp = 1 - abs(lower_bound_y - median_y) / (range + 1e-12);
        
        % Calculate objective function
        target = cov * sp;
        
        % Update maximum values
        if target > max_target_lower
            max_target_lower = target;
            max_cov_lower = cov;
            max_sp_lower = sp;
            final_lower_bound = lower_bound_y; % Update final lower bound
        end
        lower_bound_y = lower_bound_y + step_size; % Update lower bound
    end
    
    % Store results
    results(clusterIdx, :) = [max_target_upper, max_cov_upper, max_sp_upper, ...
                               max_target_lower, max_cov_lower, max_sp_lower, ...
                               final_upper_bound, final_lower_bound];
end