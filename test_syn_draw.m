close all;
clear;

%% 生成人工数据集并显示（2个x值）
% 设置随机数生成种子，确保结果可复现
rng(42);
x_min = 1;          % 输入变量的最小值
x_max = 15;         % 输入变量的最大值
% 生成 1500 对在区间 [1, 15] 内均匀分布的随机数
N = 1500;  % 数据点数
z1 = x_min + (x_max - x_min) * rand(N, 1);  % 均匀分布的随机数 z1_k
z2 = x_min + (x_max - x_min) * rand(N, 1);  % 均匀分布的随机数 z2_k

% 噪声项 epsilon，符合均值为 0，标准差为 0.5 的正态分布
sigma = 0.5;
epsilon = sigma * randn(N, 1);

% 计算 x1_k = z1_k + epsilon
x1 = z1 + epsilon;
x1(x1 < 0) = 1e-3;

% 计算 x2_k = z2_k + epsilon
x2 = z2 + epsilon;
x2(x2 < 0) = 1e-3;

% 计算 y_k = 2*sin(x1) + sqrt(x1/3) + exp(x2/25) + x2.^2/50 + epsilon
y = 2 * sin(x1) + sqrt(x1 / 3) + exp(x2 / 25) + x2.^2 / 50 + epsilon;
% y = 2 * sin(x2) + sqrt(x2 / 3) + exp(x2 / 25) + x1.^2 / 50 + epsilon;

% 组合 x1, x2 和 y
data = [x1, x2, y];


P = 3;
n_cluster = 3;
m = 2.0;
ratio = 0.3;
isOverlap = 0;
isShow = 0;
lambda = 2.2;
run_time = 1;

data_all = [x1, x2, y];
data_all = dataNormalization(data_all, 6);

% Extract normalized x1, x2, and y
x1_normalized = data_all(:,1);
x2_normalized = data_all(:,2);
y_normalized = data_all(:,3);



%% 划分train和test（2个x值）

% Combine x1 and x2 into a single feature matrix
x_features = data_all(:, 1:2);

% Extract y values
y_values = data_all(:, 3);

cv = cvpartition(size(data_all, 1), 'Holdout', 0.2);

% Split the data into training and test sets
data_train_indices = training(cv);
data_test_indices = test(cv);

data_train = [x_features(data_train_indices, :), y_values(data_train_indices)];
[n_data, n_D] = size(data_train);

test_data = [x_features(data_test_indices, :), y_values(data_test_indices)];
n_test_data = size(test_data, 1);

splitdata = split_data(data_train,P,ratio,isOverlap);


% %% 显示各端训练数据（仅2个x值）
% for i_site = 1:P
%     tmp_data = splitdata{i_site};
%     figure;
%     scatter(tmp_data(:,1), tmp_data(:,2), 'o', 'MarkerEdgeColor', [0.8,0.8,0.8], 'MarkerFaceColor', 'none');
%     xlabel('x1');
%     ylabel('x2');
%     title(['League ', num2str(i_site) ' - x1 vs x2']);
% end


%% 算法计算

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
    [center_global_R1, U_global_R1, U_global_site_R1,client_centers] = step3_global_fcm(splitdata,P,m,n_cluster,n_D,local_interval_cntrs2); 
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

% 遍历每个客户端的数据
for i_site = 1:P
    % 获取当前客户端的数据和聚类中心
    data = client_centers{i_site};
    centers = center_site_local_dpc_R1{i_site}; % 聚类中心数据

    % 创建一个新的图形窗口
    figure;
    hold on; % 允许在同一个图形上绘制多个对象

    % 遍历每个长方形的数据并绘制长方形
    for i = 1:size(data, 1)
        % 提取当前长方形的边界值
        left = data(i, 1);
        right = data(i, 2);
        bottom = data(i, 3);
        top = data(i, 4);

        % 计算位置(Position)和尺寸(Width, Height)
        pos_x = left;
        pos_y = bottom;
        width = right - left;
        height = top - bottom;

        % 绘制长方形
        rectangle('Position', [pos_x, pos_y, width, height], 'EdgeColor', 'b', 'FaceColor', 'none');
    end

    % 绘制聚类中心的散点图
    scatter(centers(:, 1), centers(:, 2), 'filled', 'r'); % 使用红色实心圆点表示聚类中心

    % 设置图形属性
    xlabel('X轴');
    ylabel('Y轴');
    title(['Client ', num2str(i_site) ' - Rectangles and Cluster Centers']);
    axis equal; % 确保x轴和y轴的比例相同，使矩形不失真
    grid on; % 显示网格线

    hold off; % 结束绘图模式
end


% %% 显示各端训练数据（原始数据、聚类中心及长方形区间）
% for i_site = 1:P
%     % 原始数据处理
%     tmp_data = splitdata{i_site};
% 
%     % 客户端的数据和聚类中心获取
%     data = client_centers{i_site};
%     centers = center_site_local_dpc_R1{i_site}; % 聚类中心数据
% 
%     % 创建一个新的图形窗口
%     figure;
%     hold on; % 允许在同一个图形上绘制多个对象
% 
%     % 绘制原始数据散点图
%     scatter(tmp_data(:,1), tmp_data(:,2), 'o', 'MarkerEdgeColor', [0.8,0.8,0.8], 'MarkerFaceColor', 'none');
% 
%     % 遍历每个长方形的数据并绘制长方形
%     for i = 1:size(data, 1)
%         % 提取当前长方形的边界值
%         left = data(i, 1);
%         right = data(i, 2);
%         bottom = data(i, 3);
%         top = data(i, 4);
% 
%         % 计算位置(Position)和尺寸(Width, Height)
%         pos_x = left;
%         pos_y = bottom;
%         width = right - left;
%         height = top - bottom;
% 
%         % 绘制长方形
%         rectangle('Position', [pos_x, pos_y, width, height], 'EdgeColor', 'b', 'FaceColor', 'none');
%     end
% 
%     % 绘制聚类中心的散点图
%     scatter(centers(:, 1), centers(:, 2), 'filled', 'r'); % 使用红色实心圆点表示聚类中心
% 
%     % 设置图形属性
%     xlabel('X1');
%     ylabel('X2');
%     title(['League ', num2str(i_site) ]);
%     axis equal; % 确保x轴和y轴的比例相同，使矩形不失真
%     grid on; % 显示网格线
% 
%     hold off; % 结束绘图模式
% end

