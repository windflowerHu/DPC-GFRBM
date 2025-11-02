function [client_prototypes, client_membership, granularity_results] = dbscan_outputs_gfrbm(splitdata, P)
    epsilon = 1;
    minPts = 2;
    
    % Initialize results
    client_prototypes = cell(1, P);  % Store prototypes for each client
    client_membership = cell(1, P); % Store membership matrix of original data to prototypes for each client
    granularity_results = cell(1, P);       % Y value intervals corresponding to prototypes for each client
    
    % DBSCAN to find prototypes for each client
    for i_site = 1:P
        tmp_data = splitdata{i_site};
        tmp_data_x = tmp_data(:, 1:end-1); % Remove last column (revenue column)
        tmp_y = tmp_data(:, end);           % Extract revenue column
        
        % Clustering using DBSCAN
        labels = dbscan(tmp_data_x, epsilon, minPts);
        
        % Filter out noise points and calculate cluster centers
        unique_labels = unique(labels(labels >= 0)); % Exclude noise points (label = -1)
        tmp_centers = zeros(length(unique_labels), size(tmp_data_x, 2));
        
        for i = 1:length(unique_labels)
            tmp_centers(i, :) = mean(tmp_data_x(labels == unique_labels(i), :), 1);
        end
        
        % Store prototypes for this client
        client_prototypes{i_site} = tmp_centers;
        
        % Calculate membership matrix
        if ~isempty(tmp_centers)
            % Calculate distance from each original data point to prototypes
            dist_matrix = pdist2(tmp_data_x, tmp_centers); % Distance from original data to cluster centers
            
            % Convert to membership (based on inverse distance)
            membership = 1 ./ (dist_matrix + epsilon); % Prevent division by zero by adding a small eps
            
            % Normalize membership
            membership = membership ./ sum(membership, 2);
            client_membership{i_site} = membership;
            
            % Call justifiable granularity principle function to calculate y value intervals
            granularity_results{i_site} = comput_gran_y([tmp_data_x, tmp_y], membership);
        else
            client_membership{i_site} = [];
            granularity_results{i_site} = [];
        end
    end
end