function [local_interval_cntrs] = step2_local_gran_dpc(splitdata, P, center_site_local_dpc_R1, U_site_local_dpc_R1, n_D)
    % Initialize a cell array to store cluster centers for each client and feature triplets for each prototype
    local_interval_cntrs = cell(P, 1);  % Initialize as P rows, number of columns dynamically adjusted later based on cluster center count
    
    % Calculate interval-valued cluster centers
    for i_site = 1:P
        Z = splitdata{i_site};
        center_site = center_site_local_dpc_R1{i_site}; % Get cluster centers for current client
        U_site = U_site_local_dpc_R1{i_site};  % Get membership matrix for current client
        Z = Z(:, 1:end-1);
        n_cluster = size(center_site, 1);
        local_interval_cntrs{i_site} = cell(n_cluster, 1);
        lower_bounds = min(Z, [], 1);
        upper_bounds = max(Z, [], 1);
        
        for c = 1:n_cluster
            v = center_site(c, :);
            local_interval_cntrs{i_site}{c} = cell(1, n_D - 1);
            
            for i = 1:n_D - 1
                if upper_bounds(i)==lower_bounds(i)
                    local_interval_cntrs{i_site}{c}{i} = {upper_bounds(i), upper_bounds(i), upper_bounds(i)};
                else
                    a_add = (v(i)-lower_bounds(i))/20;
                    b_add = (upper_bounds(i)-v(i))/20;
                    
                    if a_add == 0
                        best_a= v(i);
                    else
                        a_candi = lower_bounds(i):a_add:v(i);
                        range_a = abs(v(i) - lower_bounds(i));
                        result_a = zeros(1, numel(a_candi));
                        for a_idx = 1:numel(a_candi)
                            cov_a = sum(U_site(c, Z(:, i) >= a_candi(a_idx) & Z(:, i) <= v(i)));
                            sp_a = 1 - abs(a_candi(a_idx) - v(i)) / range_a;
                            result_a(a_idx) = cov_a * sp_a;
                        end
                        [~, idx_a_max] = max(result_a);
                        best_a = a_candi(idx_a_max);
                    end
                    
                    if b_add == 0
                        best_b = v(i);
                    else
                        b_candi = v(i):b_add:upper_bounds(i);
                        range_b = abs(upper_bounds(i) - v(i));
                        result_b = zeros(1, numel(b_candi));
                        for b_idx = 1:numel(b_candi)
                            cov_b = sum(U_site(c, Z(:, i) >= v(i) & Z(:, i) <= b_candi(b_idx)));
                            sp_b = 1 - abs(b_candi(b_idx) - v(i)) / range_b;
                            result_b(b_idx) = cov_b * sp_b;
                        end
                        [~, idx_b_max] = max(result_b);
                        best_b = b_candi(idx_b_max);
                    end
                    local_interval_cntrs{i_site}{c}{i} = {best_a, best_b, v(i)};
                end
            end
        end
    end
end