function [U_global_R1_test]= step5_gran_test(data_test,m,n_cluster,n_D,center_global_R1)
%% Substitute test data to obtain global membership matrix
% Calculate average of interval-valued prototypes to obtain global precise-valued prototypes
center_global_avg = zeros(size(center_global_R1, 1), size(center_global_R1, 2) / 2);
for i = 1:size(center_global_R1, 1)
    for j = 1:(size(center_global_R1, 2) / 2)
        min_val = center_global_R1(i, 2*j-1);
        max_val = center_global_R1(i, 2*j);
        center_global_avg(i, j) = (min_val + max_val) / 2;
    end
end

% Set FCM parameters
m = 2; % Fuzziness coefficient
max_iter = 200; % Maximum number of iterations
tol = 1e-5; % Convergence tolerance

tmp_data = data_test(:, 1:end-1);
n_samples = size(tmp_data, 1);

% Initialize membership matrix
U = rand(n_samples, n_cluster);
U = U ./ sum(U, 2);

% Iteratively update membership matrix
for iter = 1:max_iter
    % Calculate distance between data points and cluster centers
    dist = pdist2(tmp_data, center_global_avg, 'euclidean');
    
    % Update membership matrix
    U_new = zeros(size(U));
    for i = 1:n_samples
        for j = 1:n_cluster
            % Calculate membership
            sum_inv_dist = sum(1 ./ (dist(i,:) .^ (2 / (m - 1))));
            U_new(i, j) = 1 / sum((1 ./ (dist(i, :) .^ (2 / (m - 1))))) .* (1 ./ (dist(i, j) .^ (2 / (m - 1))));
        end
    end
    
    % Update membership matrix
    U = U_new;
    
    % Check convergence condition
    if max(abs(U - U_new), [], 'all') < tol
        break;
    end
end

U_global_R1_test = U;