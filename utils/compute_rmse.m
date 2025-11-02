function rmse = compute_rmse(A_flat, U, X_aug, Y, c, d)
    % Reshape flattened A back to a matrix
    A = reshape(A_flat, c, d + 1);

    % Number of data points
    N = size(X_aug, 1);

    % Initialize predicted output
    Y_hat = zeros(N, 1);

    % Compute predicted output using fuzzy inference
    for i = 1:N
        % Compute the local predictions for each cluster
        y_local = X_aug(i, :) * A'; % 1-by-c vector of local predictions
        % Weighted average of local predictions
        Y_hat(i) = sum(U(:, i) .* y_local') / sum(U(:, i));
    end

    % Calculate RMSE
    rmse = sqrt(mean((Y - Y_hat).^2));
end

% function rmse = compute_rmse(A_flat, U, X_aug, y_true, n_cluster, n_D)
%     % Reshape A_flat back to matrix form
%     A = reshape(A_flat, n_cluster, n_D);
% 
%     % Get number of data points
%     n_data = size(X_aug, 1);
% 
%     % Compute predicted outputs
%     Y_pred = zeros(n_data, 1);
% 
%     for i = 1:n_data
%         % Compute local predictions for each cluster
%         y_local = X_aug(i, :) * A'; % 1-by-c vector
% 
%         % Weighted average
%         u_sum = sum(U(:, i));
%         if u_sum > 1e-10
%             Y_pred(i) = sum(U(:, i) .* y_local') / u_sum;
%         else
%             Y_pred(i) = mean(y_local);
%         end
% 
%         % Check for invalid predictions
%         if isnan(Y_pred(i)) || isinf(Y_pred(i))
%             Y_pred(i) = mean(y_true); % Use mean as fallback
%         end
%     end
% 
%     % Compute RMSE
%     residuals = y_true - Y_pred;
%     rmse = sqrt(mean(residuals.^2));
% 
%     % Ensure RMSE is valid
%     if isnan(rmse) || isinf(rmse)
%         rmse = 1e10; % Return large penalty value
%     end
% end