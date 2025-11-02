function [client_prototypes, client_membership, granularity_results] = fcm_outputs_gfrbm(splitdata, P, n_cluster)
    fuzziness = 2;       % FCM fuzziness coefficient
    max_iter = 100;      % Maximum number of iterations
    tol = 1e-5;          % Convergence threshold
    
    % Initialize results
    client_prototypes = cell(1, P);  % Store prototypes for each client
    client_membership = cell(1, P);  % Store membership matrix of original data to prototypes for each client
    granularity_results = cell(1, P); % Y value intervals corresponding to prototypes for each client
    
    % FCM clustering
    for i_site = 1:P
        tmp_data = splitdata{i_site};
        tmp_data_x = tmp_data(:, 1:end-1); % Remove last column (revenue column)
        tmp_y = tmp_data(:, end);           % Extract revenue column
        
        % Initialize cluster centers
        initial_centers = tmp_data_x(1:n_cluster, :);  % Randomly select initial cluster centers
        
        % Use FCM clustering
        [centers, U] = FCM(tmp_data_x, n_cluster, fuzziness, max_iter, tol, initial_centers);
        
        % Store prototypes for this client
        client_prototypes{i_site} = centers;
        
        % Store membership matrix
        client_membership{i_site} = U;
        
        % Call justifiable granularity principle function to calculate y value intervals
        granularity_results{i_site} = comput_gran_y([tmp_data_x, tmp_y], U);
    end
end