clc;
clear all;
close all;

% 定义 CSV 文件的路径和文件名
filename = './data/real_income/fortune1000_2024.csv'; 


try
    % 使用 readmatrix 读取 CSV 文件
    data = readmatrix(filename, 'NumHeaderLines', 1); % 假设第一行为标题行，若无则设置为0
    
    % 截取收益特征（从第四列开始到最后）
    data = data(:, 6:end);

    % data_all = dataNormalization(data,6);

    % 最小 - 最大归一化每一列元素
    num_cols = size(data, 2);
    data_all = zeros(size(data));
    for col = 1:num_cols
        col_min = min(data(:, col));
        col_max = max(data(:, col));
        if col_max - col_min > 0
            data_all(:, col) = (data(:, col) - col_min) / (col_max - col_min);
        else
            % 如果该列所有元素相同，将该列元素都设为 0
            data_all(:, col) = zeros(size(data(:, col)));
        end
    end

    % 将 data_all 中的 NaN 项变成 0
    data_all(isnan(data_all)) = 0;

    % 划分训练集和测试集
    cv = cvpartition(size(data_all, 1), 'Holdout', 0.2); % 80% 训练集, 20% 测试集
    trainIdx = training(cv);  % 获取训练集索引
    testIdx = test(cv);       % 获取测试集索引

    % 分割训练集和测试集
    data = data_all(trainIdx, :);
    test_data = data_all(testIdx, :);

    % % 输出原始数据、训练集和测试集的信息
    % fprintf('原始数据的行数: %d, 列数: %d\n', size(data_all, 1), size(data_all, 2));
    % fprintf('训练集的行数: %d, 列数: %d\n', size(data_train, 1), size(data_train, 2));
    % fprintf('测试集的行数: %d, 列数: %d\n', size(data_test, 1), size(data_test, 2));

catch ME
    error('读取文件时出错: %s', ME.message);
end


data_i = 3;
model = 6;
m = 2.0;
ratio = 0.3;
isOverlap = 0;  % 是否允许聚类重叠
isShow = 0; 
run_time = 1;




for P = 3:3
for n_cluster = 3:3
    info_sum = {'n_cluster', n_cluster, 'P', P};
  
    for test_i = 1:1
        fprintf('test_i: %d\n',test_i);


            % 划分数据为多个端拥有
            splitdata = split_data(data,P,ratio,isOverlap);

            ground_truth_y = data(:,end);
            range_gran = max(ground_truth_y)-min(ground_truth_y);
            [n_data,n_D] = size(data);

            % test data
            ground_truth_test_y = test_data(:,end);
            n_test_data = size(test_data,1);

            % 指标初始化
            time_r1 = zeros(run_time,1); time_gran_r1 = zeros(run_time,1);
            mean_index_V_site_train = zeros(run_time,1);
            index_V_test = zeros(run_time,1);
            IA_array = zeros(run_time, 1);  % 区间精度IA  

            for i_run = 1:run_time
                tic;
                 % 1.DPC计算得到本地原型
                [center_site_local_dpc_R1, U_site_local_dpc_R1] = step1_local_dpc(splitdata, P);
                % 2.使用合理粒度原则得到本地区间值原型
                [local_interval_cntrs2] = step2_local_gran_dpc(splitdata, P, center_site_local_dpc_R1, U_site_local_dpc_R1, n_D);
                
                % 3.FCM获得全局区间值原型和全局隶属度矩阵
                [center_global_R1, U_global_R1, U_global_site_R1,all_centers] = step3_global_fcm(splitdata,P,m,n_cluster,n_D,local_interval_cntrs2); 
                time_r1(i_run) = toc;

                % 评价粒度（训练集）
                tic;
                parfor i_site = 1:P
                    data_train = splitdata{i_site};
                    U_train = U_global_site_R1{i_site};
            
                    % 4.合理粒度原则得到各个原型对应的区间值y（训练集）
                    [results_train,final_y_intervals_train] = step4_new_gran_y(data_train,U_train); 
                    % 6.评价粒度（训练集）
                    y_list_tain = data_train(:, end); % 真实值
                    y_min_values_tain = final_y_intervals_train(:, 1); % 最小值
                    y_max_values_tain = final_y_intervals_train(:, 2); % 最大值
                    % 调用 index_gran 函数，将三个列表传入
                    index_V_site_train(i_site) = index_gran(y_list_tain, y_min_values_tain, y_max_values_tain);
            
                    % disp(['index_V_site_train(i_site): ', num2str(index_V_site_train(i_site))]);
                end
                mean_index_V_site_train(i_run) = mean(index_V_site_train);
            
                time_gran_r1(i_run) = toc;
                time_gran_r1(i_run) = time_gran_r1(i_run)+ time_r1(i_run);
                disp(['研究点算法训练集的总运行时间为: ', num2str(time_gran_r1(i_run))]);
                disp(['研究点算法训练集的粒度性能评价指标为: ', num2str(mean_index_V_site_train(i_run))]);
            
                % 评价粒度（测试集）
                % 4.合理粒度原则得到各个原型对应的区间值y（测试集）
                [U_global_R1_test] = step5_gran_test(test_data,m,n_cluster,n_D,center_global_R1);  % 得到测试集的全局隶属度矩阵
                % 5.模糊推理模型计算收益值区间（测试集）
                [results,final_y_intervals] = step4_new_gran_y(test_data,U_global_R1_test);  % 得到测试集的区间值y
                % 6.评价粒度（测试集）
                y_list = test_data(:, end); % 真实值
                y_min_values = final_y_intervals(:, 1); % 最小值
                y_max_values = final_y_intervals(:, 2); % 最大值
                % 调用 index_gran 函数，将三个列表传入
                index_V_test(i_run) = index_gran(y_list, y_min_values, y_max_values);
                disp(['研究点算法测试集的粒度性能评价指标为: ', num2str(index_V_test(i_run))]);

                % 调用函数
                [IA, IS] = evaluate_intervals(final_y_intervals, y_list);
                % 将 IA 值存入 IA_array 数组
                IA_array(i_run) = IA;
                
                % 输出结果
                fprintf('研究点算法的区间精度 (IA): %.4f\n', IA);


            end



            % %% 保存old model过程数据
            % algName = 'R1';
            % type_str = 'REAL/';
            % dataname = 'syn';
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
            % time_r1_sorted = sort(roundn(time_r1, -4))';
            % max_time_r1 = max(time_r1_sorted);
            % min_time_r1 = min(time_r1_sorted);
            % avg_time_r1 = mean(time_r1_sorted);
            % std_time_r1 = std(time_r1_sorted);
            % var_time_r1 = var(time_r1_sorted);
            % 
            % % 将统计量添加到表格
            % t_time_r1 = table(time_r1_sorted, max_time_r1, min_time_r1, avg_time_r1, std_time_r1, var_time_r1, info, datetime('now'));
            % writetable(t_time_r1, path, 'Sheet', 3, 'WriteMode', 'append', 'WriteRowNames', false, 'WriteVariableNames', false);
            % 
            % % 计算粒度化运行时间统计量
            % time_gran_r1_sorted = sort(roundn(time_gran_r1, -4))';
            % max_time_gran_r1 = max(time_gran_r1_sorted);
            % min_time_gran_r1 = min(time_gran_r1_sorted);
            % avg_time_gran_r1 = mean(time_gran_r1_sorted);
            % std_time_gran_r1 = std(time_gran_r1_sorted);
            % var_time_gran_r1 = var(time_gran_r1_sorted);
            % 
            % % 将统计量添加到表格
            % t_time_gran_r1 = table(time_gran_r1_sorted, max_time_gran_r1, min_time_gran_r1, avg_time_gran_r1, std_time_gran_r1, var_time_gran_r1, info, datetime('now'));
            % writetable(t_time_gran_r1, path, 'Sheet', 4, 'WriteMode', 'append', 'WriteRowNames', false, 'WriteVariableNames', false);
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