clc;
clear all;
close all;

matrix_data_name = {'1,airfoil_self_noise','2,parkinsons_updrs','3,compactiv','4,puma32h','5,fortune1000_2024'};


data_i = 1;% Dataset number, integer not exceeding 5
normway = 1;%Data normalization method, integer from 1-4
m = 2.0;%Fuzzy parameter
ratio = 0.3; % Data overlap ratio for each site
isOverlap = 0;  % Whether to allow data distribution overlap
run_time = 20; % Number of algorithm repetitions
P = 5; % Number of sites the data will be divided into
n_cluster = 2; % Number of clusters


fprintf('data %d begain\n', data_i);
dataname = matrix_data_name{data_i};
dataname = strsplit(dataname,',');
dataname = dataname{2};
if data_i==5
    data_all = readmatrix('fortune1000_2024.csv', 'NumHeaderLines', 1);
    % Extract the features about revenue
    data_all = data_all(:, 6:end);
    data_all(isnan(data_all)) = 0;
else
    data_all = load([dataname,'.txt']);
end
data_all = dataNormalization(data_all,normway);% Data normalization

% Split into training and testing sets
cv = cvpartition(size(data_all, 1), 'Holdout', 0.2); % 80% training, 20% testing
trainIdx = training(cv); % Obtain training indices
testIdx = test(cv); % Obtain testing indices

%% Train data and partition into multiple site distributions
data = data_all(trainIdx, :);
splitdata = split_data(data,P,ratio,isOverlap);
ground_truth_y = data(:,end);
range_gran = max(ground_truth_y)- min(ground_truth_y);
[n_data,n_D] = size(data);

%% test data
test_data = data_all(testIdx, :);
n_test_data = size(test_data,1);


%% Initialize metrics    
time_cfc = zeros(run_time,1);  % Algorithm runtime
time_gran_cfc = zeros(run_time,1);  % Granulation runtime
mean_index_V_site_train = zeros(run_time,1);  % raining set performance metric
index_V_test = zeros(run_time,1); % Testing set performance metric  


for i_run = 1:run_time
    tic;
    belta = 1.2;
    sigma = 0.2;
    % training
    [V_site_center,cell_A_opt, cell_Y_hat, rmse_train_asoc, mean_rmse_train_asoc, center_site] =...
    cfc_gfrbm_prototype_train(splitdata,P,n_D,n_cluster,belta,sigma,n_data);
    % testing
    [rmse_test_asoc,mean_rmse_test_asoc,cell_Y_hat_test] = gran_result_cfc_or_iso(test_data, P, center_site, cell_A_opt, m, n_cluster);
    % Algorithm runtime
    time_cfc(i_run) = toc;

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
        index_V_site_train(i_site) = compute_V_metric(tmp_data_y,a_train,b_train);
    end
    mean_index_V_site_train(i_run) = mean(index_V_site_train);

    time_gran_cfc(i_run) = toc;
    time_gran_cfc(i_run) = time_gran_cfc(i_run)+ time_cfc(i_run);
    
    intervals_test = zeros(n_test_data, 2);
    tic;
    parfor i_data = 1:n_test_data
        y_all = cell_Y_hat_test(i_data, :);
        ym = mean(y_all);
        ymin = min(y_all);
        ymax = max(y_all);
        [lower_bound, upper_bound] = compute_PGJ_onedim(y_all, ym, ymin, ymax);  
        
        if lower_bound > upper_bound
            tmp = upper_bound;
            upper_bound = lower_bound;
            lower_bound = tmp;
        end
        
        intervals_test(i_data, :) = [lower_bound, upper_bound];
    end
    
    index_V_test(i_run) = compute_V_metric(test_data(:, end), intervals_test(:, 1), intervals_test(:, 2)); 
    
end
            
%% Save process data
algName = 'cfc_gfrbm';
folder2 = strcat('./Result/', algName, '/', num2str(data_i), '_', dataname, '/');
if ~exist(folder2, 'dir')
    mkdir(folder2);
end
path = strcat(folder2, 'analysis_result_', algName, '_', dataname, '_', num2str(normway), '_', num2str(isOverlap), '.xlsx');

% Basic information
info = {'n_cluster', n_cluster, 'P', P};

% Calculate training set performance metric statistics
mean_index_V_site_train_sorted = sort(roundn(mean_index_V_site_train, -4))';
max_train = max(mean_index_V_site_train_sorted);
min_train = min(mean_index_V_site_train_sorted);
avg_train = mean(mean_index_V_site_train_sorted);
std_train = std(mean_index_V_site_train_sorted);
var_train = var(mean_index_V_site_train_sorted);

% add statistics to table
t_train = table(mean_index_V_site_train_sorted, max_train, min_train, avg_train, std_train, var_train, info, datetime('now'));
writetable(t_train, path, 'Sheet', 1, 'WriteMode', 'append', 'WriteRowNames', false, 'WriteVariableNames', false);

% Calculate testing set performance metric statistics
index_V_test_sorted = sort(roundn(index_V_test, -4))';
max_test = max(index_V_test_sorted);
min_test = min(index_V_test_sorted);
avg_test = mean(index_V_test_sorted);
std_test = std(index_V_test_sorted);
var_test = var(index_V_test_sorted);

% Add statistics to table
t_test = table(index_V_test_sorted, max_test, min_test, avg_test, std_test, var_test, info, datetime('now'));
writetable(t_test, path, 'Sheet', 2, 'WriteMode', 'append', 'WriteRowNames', false, 'WriteVariableNames', false);

% Calculate algorithm runtime statistics
time_cfc_sorted = sort(roundn(time_cfc, -4))';
max_time_cfc = max(time_cfc_sorted);
min_time_cfc = min(time_cfc_sorted);
avg_time_cfc = mean(time_cfc_sorted);
std_time_cfc = std(time_cfc_sorted);
var_time_cfc = var(time_cfc_sorted);

% Add statistics to table
t_time_asoc = table(time_cfc_sorted, max_time_cfc, min_time_cfc, avg_time_cfc, std_time_cfc, var_time_cfc, info, datetime('now'));
writetable(t_time_asoc, path, 'Sheet', 3, 'WriteMode', 'append', 'WriteRowNames', false, 'WriteVariableNames', false);

% Calculate granulation runtime statistics
time_gran_cfc_sorted = sort(roundn(time_gran_cfc, -4))';
max_time_gran_cfc = max(time_gran_cfc_sorted);
min_time_gran_cfc = min(time_gran_cfc_sorted);
avg_time_gran_cfc = mean(time_gran_cfc_sorted);
std_time_gran_cfc = std(time_gran_cfc_sorted);
var_time_gran_cfc = var(time_gran_cfc_sorted);

% Add statistics to table
t_time_gran_cfc = table(time_gran_cfc_sorted, max_time_gran_cfc, min_time_gran_cfc, avg_time_gran_cfc, std_time_gran_cfc, var_time_gran_cfc, info, datetime('now'));
writetable(t_time_gran_cfc, path, 'Sheet', 4, 'WriteMode', 'append', 'WriteRowNames', false, 'WriteVariableNames', false);

disp('Performance metric statistics have been successfully saved to Excel');
