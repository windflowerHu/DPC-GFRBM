function [center_site_local_dpc_R1, U_site_local_dpc_R1] = step1_local_dpc(splitdata, P)
    center_site_local_dpc_R1 = cell(P, 1);  % Store cluster centers for each client
    U_site_local_dpc_R1 = cell(P, 1);       % Store membership matrix for each client
    
    % Perform DPC clustering for each client
    parfor i_site = 1:P
        data = splitdata{i_site}; % Current client's data
        data = data(:, 1:end-1);
        N = size(data, 1);
        
        % Calculate distance matrix
        distance_matrix = pdist2(data, data);
        
        % Set cutoff distance dc
        % percent = 2; % Cutoff distance is set to 2% of distances in distance matrix
        sorted_distances = sort(distance_matrix(:));
        % dc = sorted_distances(ceil(percent/100 * length(sorted_distances)));
        dc = 0.02;
        
        % Calculate local density rho for each point
        rho = sum(exp(-(distance_matrix / dc).^2), 2);
        
        % Calculate distance delta from each point to nearest higher-density point
        delta = inf(N, 1);
        for i = 1:N
            higher_density_indices = find(rho > rho(i));
            if ~isempty(higher_density_indices)
                delta(i) = min(distance_matrix(i, higher_density_indices));
            else
                delta(i) = max(distance_matrix(i, :)); % If no higher-density point exists, take maximum distance
            end
        end
        
        % Select cluster centers
        decision_values = rho .* delta;
        threshold = mean(decision_values) + std(decision_values);  % Dynamic selection of cluster centers
        center_indices = find(decision_values > threshold);
        cluster_centers = data(center_indices, :);
        num_centers = size(cluster_centers, 1);
        
        % Calculate membership matrix using Gaussian kernel function
        membership = zeros(N, num_centers);
        sigma = mean(sorted_distances);  % Kernel width is set to mean of distances
        for i = 1:N
            for j = 1:num_centers
                dist = norm(data(i, :) - cluster_centers(j, :));
                membership(i, j) = exp(-dist^2 / (2 * sigma^2));
            end
            % Normalize membership so that sum of memberships for each point equals 1
            membership(i, :) = membership(i, :) / sum(membership(i, :));
        end
        
        % Store clustering results and membership matrix for current client
        center_site_local_dpc_R1{i_site} = cluster_centers;
        U_site_local_dpc_R1{i_site} = membership;
        U_site_local_dpc_R1{i_site} = U_site_local_dpc_R1{i_site}';
    end
end