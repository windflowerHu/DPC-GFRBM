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
% n_cluster = 5;
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
        for data_i = 2:2
            % if data_i == 3|| data_i == 5 || data_i == 13 || data_i == 12|| data_i == 14|| data_i == 15
            if data_i == 3||data_i == 4|| data_i == 5 || data_i == 6 || data_i == 9 || data_i == 12 || data_i == 13|| data_i == 14|| data_i == 15|| data_i == 16
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
            range_gran = max(ground_truth_y)- min(ground_truth_y);
            % data = data(:,1:end-1);
            [n_data,n_D] = size(data);

            %% test data
            test_data = load(['./data/',type_str,'test/test_data_',dataname,'_',num2str(model),'.txt']);
            ground_truth_test_y = test_data(:,end);
            n_test_data = size(test_data,1);
            
            % 指标初始化
            rmse_train_asoc = cell(run_time,1); mean_rmse_train_asoc = zeros(run_time,1);
            rmse_test_asoc = cell(run_time,1); mean_rmse_test_asoc = zeros(run_time,1);
            
            time_asoc = zeros(run_time,1);  % 算法运行时间
            time_gran_asoc = zeros(run_time,1);  % 粒度运行时间
            mean_index_V_site_train = zeros(run_time,1);  % 训练集性能指标
            index_V_test = zeros(run_time,1); % 测试集性能指标
            IA_array = zeros(run_time, 1);  % 区间精度IA                     


            for i_run = 1:run_time
                tic;
                belta = 1.2;
                sigma = 0.2;
                % 训练集
                [V_site_center,cell_A_opt, cell_Y_hat, rmse_train_asoc, mean_rmse_train_asoc, center_site] =...
                gran_model_train(splitdata,P,n_D,n_cluster,belta,sigma,n_data);
                % 测试集
                [rmse_test_asoc,mean_rmse_test_asoc,cell_Y_hat_test] = gran_model_test(test_data, P, center_site, cell_A_opt, m, n_cluster);
                % 算法运行时间
                time_asoc(i_run) = toc;
            
                % 训练集粒度化
                tic;
                index_V_site_train = zeros(1,P);
                parfor i_site = 1:P
                    tmp_data = splitdata{i_site};
                    tmp_data_y = tmp_data(:,end);
                    tmp_data(:,end) = [];
                    tmp_n_data = length(tmp_data_y);
                    a_train = zeros(tmp_n_data,1);
                    b_train = zeros(tmp_n_data,1)
                    y_all = cell_Y_hat{i_site};  % 获取第 i_site 站点的预测值列向量
            
                    for i_data = 1:tmp_n_data
                        ym = mean(y_all);
                        ymin = min(y_all);
                        ymax = max(y_all);
                        [a_train(i_data),b_train(i_data)] = compute_PGJ_onedim(y_all,ym,ymin,ymax);  
                        if a_train(i_data) > b_train(i_data)
                            tmp_Y = b_train(i_data);
                            b_train(i_data) = a_train(i_data);
                            a_train(i_data) = tmp_Y;
                        end
                    end   
                    index_V_site_train(i_site) = index_gran(tmp_data_y,a_train,b_train);
                end
                mean_index_V_site_train(i_run) = mean(index_V_site_train);
            
                time_gran_asoc(i_run) = toc;
                time_gran_asoc(i_run) = time_gran_asoc(i_run)+ time_asoc(i_run);
                disp(['ASOC算法训练集的总运行时间为: ', num2str(time_gran_asoc(i_run))]);
                disp(['ASOC算法训练集的粒度性能评价指标为: ', num2str(mean_index_V_site_train(i_run))]);
            
                % comparison 测试集粒度化
                intervals_test = zeros(n_test_data, 2); % 每行存储 [lower_bound, upper_bound]
                tic;
                parfor i_data = 1:n_test_data
                    y_all = cell_Y_hat_test(i_data, :);
                    ym = mean(y_all);
                    ymin = min(y_all);
                    ymax = max(y_all);
                    [lower_bound, upper_bound] = compute_PGJ_onedim(y_all, ym, ymin, ymax);  
                    
                    % 确保 lower_bound <= upper_bound
                    if lower_bound > upper_bound
                        tmp = upper_bound;
                        upper_bound = lower_bound;
                        lower_bound = tmp;
                    end
                    
                    % 存储结果
                    intervals_test(i_data, :) = [lower_bound, upper_bound];
                end
                
                % 计算粒度性能指标
                index_V_test(i_run) = index_gran(test_data(:, end), intervals_test(:, 1), intervals_test(:, 2)); 
                disp(['ASOC算法测试集的粒度性能评价指标为: ', num2str(index_V_test(i_run))]);


                % 调用函数
                [IA, IS] = evaluate_intervals(intervals_test, test_data(:, end));
                % 将 IA 值存入 IA_array 数组
                IA_array(i_run) = IA;
                % 输出结果
                fprintf('ASOC算法的区间精度 (IA): %.4f\n', IA);

            end
                        
            % %% 保存old model过程数据
            % algName = 'ASOC';
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
            % time_asoc_sorted = sort(roundn(time_asoc, -4))';
            % max_time_asoc = max(time_asoc_sorted);
            % min_time_asoc = min(time_asoc_sorted);
            % avg_time_asoc = mean(time_asoc_sorted);
            % std_time_asoc = std(time_asoc_sorted);
            % var_time_asoc = var(time_asoc_sorted);
            % 
            % % 将统计量添加到表格
            % t_time_asoc = table(time_asoc_sorted, max_time_asoc, min_time_asoc, avg_time_asoc, std_time_asoc, var_time_asoc, info, datetime('now'));
            % writetable(t_time_asoc, path, 'Sheet', 3, 'WriteMode', 'append', 'WriteRowNames', false, 'WriteVariableNames', false);
            % 
            % % 计算粒度化运行时间统计量
            % time_gran_asoc_sorted = sort(roundn(time_gran_asoc, -4))';
            % max_time_gran_asoc = max(time_gran_asoc_sorted);
            % min_time_gran_asoc = min(time_gran_asoc_sorted);
            % avg_time_gran_asoc = mean(time_gran_asoc_sorted);
            % std_time_gran_asoc = std(time_gran_asoc_sorted);
            % var_time_gran_asoc = var(time_gran_asoc_sorted);
            % 
            % % 将统计量添加到表格
            % t_time_gran_asoc = table(time_gran_asoc_sorted, max_time_gran_asoc, min_time_gran_asoc, avg_time_gran_asoc, std_time_gran_asoc, var_time_gran_asoc, info, datetime('now'));
            % writetable(t_time_gran_asoc, path, 'Sheet', 4, 'WriteMode', 'append', 'WriteRowNames', false, 'WriteVariableNames', false);
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