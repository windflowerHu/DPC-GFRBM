function [client_prototypes, client_membership, granularity_results] = df_outputs_gfrbm(splitdata, P, n_cluster)
    epsilon = 1;
    minPts = 2;
    fuzziness = 2.0; % Fuzziness coefficient for FCM
    max_iter = 100; % Maximum iterations for FCM
    tol = 1e-5; % Convergence threshold for FCM
    
    % Initialize results
    client_prototypes = cell(P, 1);  % Store cluster centers for each client
    client_membership = cell(P, 1); % Store membership matrix of original data to cluster centers for each client
    granularity_results = cell(P, 1); % Y value intervals corresponding to cluster centers for each client
    
    % Iterate through each client's data
    for i_site = 1:P
        tmp_data = splitdata{i_site};
        tmp_data_x = tmp_data(:, 1:end-1); % Remove last column (revenue column)
        tmp_y = tmp_data(:, end);          % Extract revenue column
        
        % Use DBSCAN clustering to partition original data into multiple small datasets
        labels = dbscan(tmp_data_x, epsilon, minPts);
        
        % Get all valid cluster labels
        unique_labels = unique(labels(labels >= 0)); % Exclude noise points (label = -1)
        num_dbscan_clusters = length(unique_labels);
        
        % % Output DBSCAN cluster count
        % disp(['Client ', num2str(i_site), ' DBSCAN cluster count: ', num2str(num_dbscan_clusters)]);
        
        % Initialize results for this client
        client_prototypes{i_site} = []; % Store merged cluster centers
        client_membership{i_site} = []; % Store merged membership matrix
        granularity_results{i_site} = []; % Store merged granularity results
        
        % Iterate through each DBSCAN cluster subset
        for i_cluster = 1:num_dbscan_clusters
            % Extract data belonging to current subset
            sub_data_idx = labels == unique_labels(i_cluster);
            sub_data_x = tmp_data_x(sub_data_idx, :);
            sub_data_y = tmp_y(sub_data_idx);
            
            % Skip if sub-dataset is empty
            if isempty(sub_data_x)
                continue;
            end
            
            % Randomly select data points from subset as initial cluster centers
            num_points = size(sub_data_x, 1);
            if num_points < n_cluster
                % If insufficient data points to initialize all centers, randomly repeat data points
                init_centers = sub_data_x(randi(num_points, n_cluster, 1), :);
            else
                % Randomly select non-repeating initial centers
                init_centers = sub_data_x(randperm(num_points, n_cluster), :);
            end
            
            % Use FCM to cluster the sub-dataset
            [fcm_centers, fcm_membership] = FCM(sub_data_x, n_cluster, fuzziness, max_iter, tol, init_centers);
            
            % Merge FCM cluster centers corresponding to this DBSCAN cluster
            client_prototypes{i_site} = [client_prototypes{i_site}; fcm_centers];
            
            % % Output FCM cluster count
            % disp(['Client ', num2str(i_site), ' FCM cluster count (subset ', num2str(i_cluster), '): ', num2str(n_cluster)]);
            
            % Calculate membership matrix for this sub-dataset
            dist_matrix = pdist2(tmp_data_x, fcm_centers); % Distance from original data to new cluster centers
            sub_membership = 1 ./ (dist_matrix + epsilon); % Convert to membership (prevent division by zero)
            sub_membership = sub_membership ./ sum(sub_membership, 2); % Normalize
            
            % Merge membership matrix corresponding to this DBSCAN cluster
            client_membership{i_site} = [client_membership{i_site}, sub_membership];
            
            % Store granularity results for this DBSCAN cluster
            for j = 1:n_cluster
                granularity_result = comput_gran_y([tmp_data_x, tmp_y], sub_membership(:, j));
                % Merge granularity results for each FCM cluster center into granularity_results
                granularity_results{i_site} = [granularity_results{i_site}; granularity_result];
            end
        end
    end
end