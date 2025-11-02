function [center_global_R1, U_global_R1,U_global_site_R1,client_centers]= step3_global_fcm(splitdata,P,m,n_cluster,n_D,local_interval_cntrs)
%% Obtain local interval-valued prototypes
% Initialize a cell array to store the number of cluster centers for each client
cluster_counts = zeros(P, 1);
for i = 1:P
    cluster_counts(i) = size(local_interval_cntrs{i}, 1); % Get number of clusters for each client
end

% Calculate total number of cluster centers
total_clusters = sum(cluster_counts);

% Initialize a 2D array to store minimum and maximum values of features for each prototype
all_centers = zeros(total_clusters, (n_D - 1) * 2);

% Initialize a cell array to store cluster centers by client
client_centers = cell(P, 1);

% Extract minimum and maximum values and reorganize data
index = 1;
for i = 1:P
    % Initialize temporary matrix based on current client's cluster center count
    client_data = zeros(cluster_counts(i), (n_D - 1) * 2);
    
    for j = 1:cluster_counts(i)
        center = local_interval_cntrs{i}{j};
        
        % Extract minimum and maximum values for each feature and form binary tuples
        feature_vector = zeros(1, (n_D - 1) * 2);
        for k = 1:n_D - 1
            min_val = center{k}{1};
            max_val = center{k}{2};
            feature_vector(2*k - 1) = min_val;
            feature_vector(2*k) = max_val;
        end
        
        all_centers(index, :) = feature_vector;
        client_data(j, :) = feature_vector;
        index = index + 1;
    end
    client_centers{i} = client_data;
end

%% Secondary clustering to obtain global interval-valued prototypes
tmp_n_data = size(all_centers,1);
tmp_indx = randperm(tmp_n_data, n_cluster);
center_ini = all_centers(tmp_indx,:);
[center_global_R1] = FCM(all_centers, n_cluster, m, 200, 1e-5, center_ini);

%% Substitute original data to obtain global membership matrix
% Set FCM parameters
m = 2; % Fuzziness coefficient
max_iter = 200; % Maximum number of iterations
tol = 1e-5; % Convergence tolerance

U_global_site_R1 = cell(P, 1);
for i_site = 1:P
    tmp_data = splitdata{i_site};
    tmp_data = tmp_data(:, 1:end-1);
    n_samples = size(tmp_data, 1);
    
    % Initialize membership matrix
    U = rand(n_samples, n_cluster);
    U = U ./ sum(U, 2);
    
    % Combined distance of midpoint and boundaries
    dist_combined = zeros(n_samples, n_cluster);
    for i = 1:n_samples
        for j = 1:n_cluster
            center_avg = mean([center_global_R1(j, 1:2:end); center_global_R1(j, 2:2:end)], 1); % Interval midpoint
            min_vals = center_global_R1(j, 1:2:end);
            max_vals = center_global_R1(j, 2:2:end);
            dist_combined(i, j) = 0.5 * (norm(tmp_data(i, :) - center_avg) + min(norm(tmp_data(i, :) - min_vals), norm(tmp_data(i, :) - max_vals)));
        end
    end
    dist = dist_combined;
    
    % Iteratively update membership matrix
    for iter = 1:max_iter
        U_new = zeros(size(U));
        for i = 1:n_samples
            for j = 1:n_cluster
                sum_inv_dist = sum(1 ./ (dist(i,:) .^ (2 / (m - 1))));
                U_new(i, j) = (1 / dist(i, j) .^ (2 / (m - 1))) / sum_inv_dist;
            end
        end
        
        if max(abs(U - U_new), [], 'all') < tol
            break;
        end
        U = U_new;
    end
    U_global_site_R1{i_site} = U;
end

% Vertically concatenate each U_global_site_R1 into a large matrix U_global_R1
U_global_R1 = vertcat(U_global_site_R1{:});