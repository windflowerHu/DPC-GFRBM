clc;
clear all;
close all;

matrix_data_1 = {'1,S1'};
matrix_data_2 = {'1,airfoil_self_noise','2,parkinsons_updrs','3,residential_building','4,wine_quality','5,yacht_hydrodynamics',...
    '6,CBM','7,compactiv','8,puma32h','9,elevators','10,ailerons','11,AirQuality','12,blog',...
    '13,wave','14,superconduct','15,MV','16,facebook','17,onlineNews','18,abalone'};


datatype = 2;
if datatype==1
    matrix_data = matrix_data_1;
    model_start = 8;
    type_str = 'Synthe/';
elseif datatype==2
    matrix_data = matrix_data_2;
    model_start = 1;
    type_str = 'UCI/';
end

% data_i = 6;
model = 6;
% P = 3;
% n_cluster = 6;
m = 2.0;
ratio = 0.3;
isOverlap = 0;  % 是否允许聚类重叠
isShow = 0;
run_time = 10;

for P = 3:3
for n_cluster = 2:2
    info_sum = {'n_cluster', n_cluster, 'P', P};
 
    for test_i = 1:1
        fprintf('test_i: %d\n',test_i); 
        for data_i = 8:8
            % if  data_i == 3|| data_i == 5 || data_i == 13 || data_i == 12|| data_i == 14|| data_i == 15|| data_i == 8|| data_i == 17
            if data_i == 3||data_i == 4|| data_i == 5 || data_i == 6 || data_i == 9 || data_i == 12 || data_i == 13|| data_i == 14|| data_i == 15|| data_i == 16|| data_i == 17
                continue;
            end
        fprintf('data %d begain\n', data_i);
        dataname = matrix_data{data_i};
        dataname = strsplit(dataname,',');
        dataname = dataname{2};
        data = load(['./data/',type_str,'train/train_data_',dataname,'_',num2str(model),'.txt']);
        
        %% 划分数据为多个端拥有
        splitdata = split_data(data,P,ratio,isOverlap);
        
        ground_truth_y = data(:,end);
        range_gran =max(ground_truth_y)-min(ground_truth_y);
        % data = data(:,1:end-1);
        [n_data,n_D] = size(data);
        
        %% test data
        test_data = load(['./data/',type_str,'test/test_data_',dataname,'_',num2str(model),'.txt']);
        ground_truth_test_y = test_data(:,end);
        n_test_data = size(test_data,1);
        
        % 指标初始化
        time_dbscan = zeros(run_time,1);
        time_gran_dbscan =zeros(run_time,1);
        mean_index_V_site_train = zeros(run_time,1);
        index_V_test = zeros(run_time,1);
        IA_array = zeros(run_time, 1);  % 区间精度IA 
        
        for i_run = 1:run_time
            tic;
         
            % model1:DBSCAN
            [client_prototypes, client_membership, granularity_results] = dbscan_model1(splitdata, P);

            % model2:DBSCAN+FCM
            % [client_prototypes, client_membership, granularity_results] = dbscan_model2(splitdata, P, n_cluster);

            % model3:FCM
            % [client_prototypes, client_membership, granularity_results] = dbscan_model3(splitdata, P,n_cluster);

            time_dbscan(i_run) = toc;

            % 评价粒度（训练集）
            tic;
            parfor i_site = 1:P
                data_train = splitdata{i_site};


                % % 4.合理粒度原则得到各个原型对应的区间值y（训练集）
                % [results_train] = step4_gran_y(data_train(:, 1:end-1),U_train);

                % 5.模糊推理模型计算收益值区间（训练集）
                upper_bounds_train = granularity_results{i_site}(:, 7); % 提取上界
                lower_bounds_train = granularity_results{i_site}(:, 8); % 提取下界
                bounds_matrix_train = [upper_bounds_train, lower_bounds_train];

                % disp(['bounds_matrix_train ', num2str(i_site), ' : ', num2str(size(bounds_matrix_train))]);
                % disp(['client_membership{i_site} ', num2str(i_site), ' : ', num2str(size(client_membership{i_site}))]);

                y_intervals_train = client_membership{i_site} * bounds_matrix_train; 
                % 输出每条数据的 y 值的最小值和最大值
                y_values_train = [y_intervals_train(:, 2), y_intervals_train(:, 1)]; % 每行是 [最小值, 最大值]
                % 6.评价粒度（训练集）
                y_list_tain = data_train(:, end); % 真实值
                y_min_values_tain = y_values_train(:, 1); % 最小值
                y_max_values_tain = y_values_train(:, 2); % 最大值
                % 调用 index_gran 函数，将三个列表传入
                index_V_site_train(i_site) = index_gran(y_list_tain, y_min_values_tain, y_max_values_tain);

                % disp(['index_V_site_train(i_site): ', num2str(index_V_site_train(i_site))]);
            end
            mean_index_V_site_train(i_run) = mean(index_V_site_train);

            time_gran_dbscan(i_run) = toc;
            time_gran_dbscan(i_run) = time_gran_dbscan(i_run)+ time_dbscan(i_run);
            disp(['DBSCAN算法训练集的总运行时间为: ', num2str(time_gran_dbscan(i_run))]);
            disp(['DBSCAN算法训练集的粒度性能评价指标为: ', num2str(mean_index_V_site_train(i_run))]);

            % 评价粒度（测试集）
            % 初始化
            num_test_points = size(test_data, 1);
            y_intervals_clients = zeros(num_test_points, 2, P); % 第三维是客户端编号
            
            % 遍历每个客户端，计算隶属度矩阵和区间值
            for i_site = 1:P
                % 当前客户端的聚类中心
                client_center = client_prototypes{i_site}; % 客户端的聚类中心
            
                % 计算数据点到客户端聚类中心的距离矩阵
                tmp_data = test_data(:, 1:end-1); % 去掉最后一列
                dist_data_center = pdist2(tmp_data, client_center);
            
                % 防止除零
                dist_data_center(dist_data_center == 0) = 1e-6;
            
                % 模糊化参数
                uf = -2 / (m - 1);
            
                % 计算隶属度矩阵
                U_client_test = (dist_data_center .^ uf) ./ (sum(dist_data_center .^ uf, 2) * ones(1, size(client_center, 1)));
            
                % 提取当前客户端的区间值边界
                upper_bounds_site = granularity_results{i_site}(:, 7); % 提取上界
                lower_bounds_site = granularity_results{i_site}(:, 8); % 提取下界
                bounds_matrix_site = [upper_bounds_site, lower_bounds_site];
            
                % 计算该客户端对测试集的区间值
                y_intervals_site = U_client_test * bounds_matrix_site;
            
                % 存储每个客户端的区间值
                y_intervals_clients(:, :, i_site) = [y_intervals_site(:, 2), y_intervals_site(:, 1)]; % 每行是 [最小值, 最大值]
            end
            
            % 对所有客户端的区间值取平均，得到最终的区间值
            y_intervals_final = mean(y_intervals_clients, 3); % 在客户端维度上取平均
            y_values = [y_intervals_final(:, 1), y_intervals_final(:, 2)]; % 每行是 [最小值, 最大值]
            
            % 评价粒度（测试集）
            y_list = test_data(:, end); % 真实值
            y_min_values = y_values(:, 1); % 最小值
            y_max_values = y_values(:, 2); % 最大值

            % 调用 index_gran 函数，将三个列表传入
            index_V_test(i_run) = index_gran(y_list, y_min_values, y_max_values);

            % 输出结果
            disp(['DBSCAN算法测试集的粒度性能评价指标为: ', num2str(index_V_test(i_run))]);

             % 调用函数
            [IA, IS] = evaluate_intervals(y_values, y_list);
            % 将 IA 值存入 IA_array 数组
            IA_array(i_run) = IA;
            % 输出结果
            fprintf('DBSCAN算法的区间精度 (IA): %.4f\n', IA);


        end

        % %% 保存old model过程数据
        % algName = 'DBSCAN_MODEL2';
        % folder2 = strcat('./Analysis/', algName, '/', type_str, num2str(data_i), '_', dataname, '/');
        % if ~exist(folder2, 'dir')
        %     mkdir(folder2);
        % end
        % path = strcat(folder2, 'analysis_result_', algName, '_', dataname, '_', num2str(model), '_', num2str(isOverlap), '.xlsx');
        % 
        % % 基础信息
        % info = {'n_cluster', n_cluster, 'P', P}; % 训练集指标
        % 
        % % 计算训练集性能指标统计量
        % mean_index_V_site_train_sorted = sort(roundn(mean_index_V_site_train, -4))';
        % max_train = max(mean_index_V_site_train_sorted);
        % min_train = min(mean_index_V_site_train_sorted);
        % avg_train = mean(mean_index_V_site_train_sorted);
        % std_train = std(mean_index_V_site_train_sorted);
        % var_train = var(mean_index_V_site_train_sorted);
        % 
        % % 将统计量添加到表格
        % t_train = table(mean_index_V_site_train_sorted, max_train, min_train, avg_train, std_train, var_train, info, datetime('now'));
        % writetable(t_train, path, 'Sheet', 1, 'WriteMode', 'append', 'WriteRowNames', false, 'WriteVariableNames', false);
        % 
        % % 计算测试集性能指标统计量
        % index_V_test_sorted = sort(roundn(index_V_test, -4))';
        % max_test = max(index_V_test_sorted);
        % min_test = min(index_V_test_sorted);
        % avg_test = mean(index_V_test_sorted);
        % std_test = std(index_V_test_sorted);
        % var_test = var(index_V_test_sorted);
        % 
        % % 将统计量添加到表格
        % t_test = table(index_V_test_sorted, max_test, min_test, avg_test, std_test, var_test, info, datetime('now'));
        % writetable(t_test, path, 'Sheet', 2, 'WriteMode', 'append', 'WriteRowNames', false, 'WriteVariableNames', false);
        % 
        % % 计算算法运行时间统计量
        % time_dbscan_sorted = sort(roundn(time_dbscan, -4))';
        % max_time_dbscan = max(time_dbscan_sorted);
        % min_time_dbscan = min(time_dbscan_sorted);
        % avg_time_dbscan = mean(time_dbscan_sorted);
        % std_time_dbscan = std(time_dbscan_sorted);
        % var_time_dbscan = var(time_dbscan_sorted);
        % 
        % % 将统计量添加到表格
        % t_time_dbscan = table(time_dbscan_sorted, max_time_dbscan, min_time_dbscan, avg_time_dbscan, std_time_dbscan, var_time_dbscan, info, datetime('now'));
        % writetable(t_time_dbscan, path, 'Sheet', 3, 'WriteMode', 'append', 'WriteRowNames', false, 'WriteVariableNames', false);
        % 
        % % 计算粒度化运行时间统计量
        % time_gran_dbscan_sorted = sort(roundn(time_gran_dbscan, -4))';
        % max_time_gran_dbscan = max(time_gran_dbscan_sorted);
        % min_time_gran_dbscan = min(time_gran_dbscan_sorted);
        % avg_time_gran_dbscan = mean(time_gran_dbscan_sorted);
        % std_time_gran_dbscan = std(time_gran_dbscan_sorted);
        % var_time_gran_dbscan = var(time_gran_dbscan_sorted);
        % 
        % % 将统计量添加到表格
        % t_time_gran_dbscan = table(time_gran_dbscan_sorted, max_time_gran_dbscan, min_time_gran_dbscan, avg_time_gran_dbscan, std_time_gran_dbscan, var_time_gran_dbscan, info, datetime('now'));
        % writetable(t_time_gran_dbscan, path, 'Sheet', 4, 'WriteMode', 'append', 'WriteRowNames', false, 'WriteVariableNames', false);
        % 
        % % 计算区间精度统计量
        % IA_array_sorted = sort(roundn(IA_array, -4))';
        % max_IA = max(IA_array_sorted);
        % min_IA = min(IA_array_sorted);
        % avg_IA = mean(IA_array_sorted);
        % std_IA = std(IA_array_sorted);
        % var_IA = var(IA_array_sorted);
        % 
        % % 将统计量添加到表格
        % t_IA = table(IA_array_sorted, max_IA, min_IA, avg_IA, std_IA, var_IA, info, datetime('now'));
        % writetable(t_IA, path, 'Sheet', 5, 'WriteMode', 'append', 'WriteRowNames', false, 'WriteVariableNames', false);
        % 
        % disp('性能指标统计量已成功保存到 Excel 中');
        end
    end
end
end