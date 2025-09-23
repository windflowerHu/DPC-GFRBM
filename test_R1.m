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
    % for percent = 2:2
for n_cluster = 2:2
    info_sum = {'n_cluster', n_cluster, 'P', P};
  
    for test_i = 1:1
        fprintf('test_i: %d\n',test_i);
        for data_i = 2:2
            % if data_i == 3 || data_i == 5 || data_i == 13 || data_i == 12 || data_i == 14 || data_i == 15
            if data_i == 3||data_i == 4|| data_i == 5 || data_i == 6 || data_i == 9 || data_i == 12 || data_i == 13|| data_i == 14|| data_i == 15|| data_i == 16
                continue;
            end
            fprintf('data %d begain\n', data_i);
            dataname = matrix_data{data_i};
            dataname = strsplit(dataname,',');
            dataname = dataname{2};
            data = load(['./data/',type_str,'train/train_data_',dataname,'_',num2str(model),'.txt']);

            % 划分数据为多个端拥有
            splitdata = split_data(data,P,ratio,isOverlap);

            ground_truth_y = data(:,end);
            range_gran = max(ground_truth_y)-min(ground_truth_y);
            [n_data,n_D] = size(data);

            % test data
            test_data = load(['./data/',type_str,'test/test_data_',dataname,'_',num2str(model),'.txt']);
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
                [center_global_R1, U_global_R1, U_global_site_R1] = step3_global_fcm(splitdata,P,m,n_cluster,n_D,local_interval_cntrs2); 
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
            % % algName = 'R1_FCM';
            % folder2 = strcat('./Result2/', algName, '/', type_str, num2str(data_i), '_', dataname, '/');
            % if ~exist(folder2, 'dir')
            %     mkdir(folder2);
            % end
            % path = strcat(folder2, 'process_result_', algName, '_', dataname, '_', num2str(model), '_', num2str(isOverlap), '.xlsx');
            % 
            % % 基础信息
            % info = {'n_cluster', n_cluster, 'P', P}; % 训练集指标
            % 
            % % 训练集性能指标
            % t = table(sort(roundn(mean_index_V_site_train,-4))',info,datetime('now'));
            % writetable(t,path,Sheet=1,WriteMode='append',WriteRowNames=false,WriteVariableNames=false);
            % % 测试集性能指标
            % t = table(sort(roundn(index_V_test,-4))',info,datetime('now'));
            % writetable(t,path,Sheet=2,WriteMode='append',WriteRowNames=false,WriteVariableNames=false);
            % % 算法运行时间
            % t = table(sort(roundn(time_r1,-4))',info,datetime('now'));
            % writetable(t,path,Sheet=3,WriteMode='append',WriteRowNames=false,WriteVariableNames=false);
            % % 粒度化运行时间
            % t = table(sort(roundn(time_gran_r1,-4))',info,datetime('now'));
            % writetable(t,path,Sheet=4,WriteMode='append',WriteRowNames=false,WriteVariableNames=false);
            % 
            % fprintf('R1 model end\n');
            % fprintf('Data %d end\n', data_i);
            % 
            % % 区间精度
            % t = table(sort(roundn(IA,-4))',info,datetime('now'));
            % writetable(t,path,Sheet=5,WriteMode='append',WriteRowNames=false,WriteVariableNames=false);
            % % 区间得分
            % t = table(sort(roundn(IS,-4))',info,datetime('now'));
            % writetable(t,path,Sheet=6,WriteMode='append',WriteRowNames=false,WriteVariableNames=false);
            % 
 
            % %% 保存old model过程数据
            % algName = 'R1';
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
    % end
end