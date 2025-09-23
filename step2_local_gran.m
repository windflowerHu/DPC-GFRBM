% function [local_interval_cntrs] = step2_local_gran(splitdata, P, n_cluster, center_site_local_R1, U_site_local_R1, n_D)
% 
%     % 初始化一个cell数组用于存储每个原型的特征三元组
%     local_interval_cntrs = cell(P, n_cluster);
%     % 对每个客户端和每个聚类中心进行初始化
%     for i = 1:P
%         for j = 1:n_cluster
%             % 初始化每个元素为包含n_D个元素的空元组
%             local_interval_cntrs{i,j} = cell(1, n_D - 1);
%         end
%     end
% 
%     alpha = 1; % 特异性参数
% 
%     % 计算区间值聚类中心
%     for i_site = 1:P
%         Z = splitdata{i_site}; % 从splitdata获取数据
%         center_site = center_site_local_R1{i_site}; % 获取当前客户端的聚类中心（维度为聚类数量n_cluster*特征数x）
%         U_site = U_site_local_R1{i_site}; % 获取当前客户端的隶属度矩阵（维度为聚类数量n_cluster*数据量n）
%         % disp(['Dimensions of center_site for client ', num2str(i_site), ': ', num2str(size(center_site))]);
%         % disp(['Dimensions of U_site for client ', num2str(i_site), ': ', num2str(size(U_site))]);
% 
%         Z = Z(:, 1:end-1); % 去除最后一列（y）
%         interval_cntrs = []; % 区间值原型
% 
%         % 初始化存储上下界的向量
%         lower_bounds = zeros(1, size(Z, 2)); % 存储特征的下界
%         upper_bounds = zeros(1, size(Z, 2)); % 存储特征的上界
% 
%         % 遍历每个特征列
%         for feature_idx = 1:size(Z, 2)
%             % 获取当前特征列
%             feature_values = Z(:, feature_idx);
% 
%             % 找到当前特征列的下界和上界
%             lower_bounds(feature_idx) = min(feature_values);
%             upper_bounds(feature_idx) = max(feature_values);
%         end
% 
%         % disp('Lower Bounds of Features:');
%         % disp(lower_bounds);
%         % disp('Upper Bounds of Features:');
%         % disp(upper_bounds);
% 
%         % 初始化存储每个原型下界的向量
%         all_best_a = zeros(n_cluster, numel(lower_bounds));
% 
% 
%         for c = 1:n_cluster 
%             v = center_site(c, :);  % 第 c 个原型（包含了 n 个特征）
% 
%             % 遍历每个特征
%             for i = 1:n_D-1
%                 a_candi = lower_bounds(i):0.05:v(i);  % 下界范围
%                 b_candi = v(i):0.05:upper_bounds(i);  % 上界范围
%                 range_a = abs(v(i) - lower_bounds(i)); 
%                 range_b = abs(upper_bounds(i) - v(i));
% 
%                 result_a = zeros(1, numel(a_candi)); % 存储每个 a 取值的结果
%                 result_b = zeros(1, numel(b_candi)); % 存储每个 b 取值的结果
% 
%                 for a_idx = 1:numel(a_candi)
%                     % 初始化隶属度叠加
%                     cov_a = 0;  % 存储隶属度叠加值，初始化为0
% 
%                     % 遍历 Z 的每一行
%                     for row = 1:size(Z, 1)
%                         % 如果 Z(row, i) 的取值在 a_candi 到 v(i) 之间
%                         if Z(row, i) >= a_candi(a_idx) && Z(row, i) <= v(i)
%                             % 输出 U_site 的维度
%                             % disp(['Dimensions of U_site: ', num2str(size(U_site))]);
%                             % 叠加隶属度
%                             cov_a = cov_a + U_site(i, row);
%                         end
%                     end
% 
%                     % 计算特异性
%                     sp_a = 1 - abs(a_candi(a_idx) - v(i)) / range_a;
%                     % euclidean_distance = norm(v(i) - a_candi(a_idx));
%                     % sp_a = exp(-alpha * euclidean_distance); 
% 
%                     % 存储乘积结果
%                     result_a(a_idx) = cov_a * sp_a;
%                 end
% 
%                 for b_idx = 1:numel(b_candi)
%                     % 初始化隶属度叠加
%                     cov_b = 0;  % 存储隶属度叠加值，初始化为0
%                     % 遍历 Z 的每一行
%                     for row = 1:size(Z, 1)
%                         if  Z(row, i) >= v(i) && Z(row, i) <= b_candi(b_idx)
%                             cov_b = cov_b + U_site(i, row);
%                         end
%                     end
% 
%                     % 计算特异性
%                     sp_b = 1 - abs(b_candi(b_idx) - v(i)) / range_b;
%                     % euclidean_distance = norm(b_candi(b_idx) - v(i));
%                     % sp_b = exp(-alpha * euclidean_distance); 
% 
%                     % 存储乘积结果
%                     result_b(b_idx) = cov_b * sp_b;
%                 end
% 
%                 % 找到乘积最大的情况下的 best_a 和 best_b
%                 [~, idx_a_max] = max(result_a);
%                 best_a = a_candi(idx_a_max);
% 
%                 [~, idx_b_max] = max(result_b);
%                 best_b = b_candi(idx_b_max);
%                 % count_out = counts(idx);
% 
% 
%                 % 存储特征的下界、上界和精确值到cell数组中
%                 local_interval_cntrs{i_site, c}{i} = {best_a, best_b, v(i)};
%                 % disp(['Feature triplet for prototype ', num2str(c)]);
%                 % disp(local_interval_cntrs{i_site, c}{i});
%                 % disp(['Size of feature triplet: ', num2str(size(local_interval_cntrs{i_site, c}{i}))]);
% 
%             end
% 
%         end
% 
%     end
% % disp(['Size of local_interval_cntrs: ', num2str(size(local_interval_cntrs))]);
% 
% 
% % 优化方向：原型粒度太大
% 
% 
% 

function [local_interval_cntrs] = step2_local_gran(splitdata, P, center_site_local_R1, U_site_local_R1, n_D)

    % 初始化一个cell数组用于存储每个客户端的聚类中心和每个原型的特征三元组
    local_interval_cntrs = cell(P, 1); % 初始化为P行，1列
    
    alpha = 1; % 特异性参数

    % 计算区间值聚类中心
    for i_site = 1:P
        Z = splitdata{i_site}; % 从splitdata获取数据
        center_site = center_site_local_R1{i_site}; % 获取当前客户端的聚类中心
        U_site = U_site_local_R1{i_site}; % 获取当前客户端的隶属度矩阵
        
        Z = Z(:, 1:end-1); % 去除最后一列（y）

        % 获取当前客户端的聚类数量
        n_cluster = size(center_site, 1); % 动态获取聚类数量
        local_interval_cntrs{i_site} = cell(n_cluster, 1); % 初始化为n_cluster行，1列
        
        % 初始化存储上下界的向量
        lower_bounds = zeros(1, size(Z, 2)); % 存储特征的下界
        upper_bounds = zeros(1, size(Z, 2)); % 存储特征的上界
        
        % 遍历每个特征列
        for feature_idx = 1:size(Z, 2)
            % 获取当前特征列
            feature_values = Z(:, feature_idx);
            
            % 找到当前特征列的下界和上界
            lower_bounds(feature_idx) = min(feature_values);
            upper_bounds(feature_idx) = max(feature_values);
        end

        % 初始化存储每个原型下界的向量
        all_best_a = zeros(n_cluster, numel(lower_bounds));

        for c = 1:n_cluster 
            v = center_site(c, :);  % 第 c 个原型（包含了 n 个特征）
            
            % 遍历每个特征
            for i = 1:n_D-1
                a_candi = lower_bounds(i):0.05:v(i);  % 下界范围
                b_candi = v(i):0.05:upper_bounds(i);  % 上界范围
                range_a = abs(v(i) - lower_bounds(i)); 
                range_b = abs(upper_bounds(i) - v(i));
                
                result_a = zeros(1, numel(a_candi)); % 存储每个 a 取值的结果
                result_b = zeros(1, numel(b_candi)); % 存储每个 b 取值的结果

                for a_idx = 1:numel(a_candi)
                    % 初始化隶属度叠加
                    cov_a = 0;  % 存储隶属度叠加值，初始化为0
                    
                    % 遍历 Z 的每一行
                    for row = 1:size(Z, 1)
                        % 如果 Z(row, i) 的取值在 a_candi 到 v(i) 之间
                        if Z(row, i) >= a_candi(a_idx) && Z(row, i) <= v(i)
                            cov_a = cov_a + U_site(c, row); % 使用c而不是i
                        end
                    end
                    
                    % 计算特异性
                    sp_a = 1 - abs(a_candi(a_idx) - v(i)) / range_a;
                    % 存储乘积结果
                    result_a(a_idx) = cov_a * sp_a;
                end

                for b_idx = 1:numel(b_candi)
                    % 初始化隶属度叠加
                    cov_b = 0;  % 存储隶属度叠加值，初始化为0
                    % 遍历 Z 的每一行
                    for row = 1:size(Z, 1)
                        if  Z(row, i) >= v(i) && Z(row, i) <= b_candi(b_idx)
                            cov_b = cov_b + U_site(c, row); 
                        end
                    end
                    
                    % 计算特异性
                    sp_b = 1 - abs(b_candi(b_idx) - v(i)) / range_b;
                    % 存储乘积结果
                    result_b(b_idx) = cov_b * sp_b;
                end
                
                % 找到乘积最大的情况下的 best_a 和 best_b
                [~, idx_a_max] = max(result_a);
                best_a = a_candi(idx_a_max);

                [~, idx_b_max] = max(result_b);
                best_b = b_candi(idx_b_max);
                
                % 存储特征的下界、上界和精确值到cell数组中
                local_interval_cntrs{i_site}{c}{i} = {best_a, best_b, v(i)};
            end

        end

    end
end

