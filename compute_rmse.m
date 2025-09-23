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