clc;
clear all;
close all;

matrix_data_name = {'1,airfoil_self_noise','2,parkinsons_updrs','3,compactiv','4,puma32h','5,fortune1000_2024'};


data_i = 5;% Dataset number, integer not exceeding 5
normway = 2;%Data normalization method, integer from 1-4
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
time_dpc = zeros(run_time,1);  % Algorithm runtime
time_gran_dpc = zeros(run_time,1);  % Granulation runtime
mean_index_V_site_train = zeros(run_time,1);  % raining set performance metric
index_V_test = zeros(run_time,1); % Testing set performance metric                 
 

for i_run = 1:run_time
    tic;
    % 1. DPC calculates local prototypes
    [center_site_local_dpc_R1, U_site_local_dpc_R1] = step1_local_dpc(splitdata, P);
    
    % 2. Use justifiable granularity principle to obtain local interval-valued prototypes
    [local_interval_cntrs2] = step2_local_gran_dpc(splitdata, P, center_site_local_dpc_R1, U_site_local_dpc_R1, n_D);
    
    % 3. FCM obtains global interval-valued prototypes and global membership matrix
    [center_global_R1, U_global_R1, U_global_site_R1] = step3_global_fcm(splitdata,P,m,n_cluster,n_D,local_interval_cntrs2);
    time_dpc(i_run) = toc;
    
    tic;
    parfor i_site = 1:P
        data_train = splitdata{i_site};
        U_train = U_global_site_R1{i_site};
        
        % 4. Justifiable granularity principle obtains interval-valued y corresponding to each prototype (training set)
        [results_train,final_y_intervals_train] = step4_new_gran_y(data_train,U_train);
        
        % 6. Calculate granularity metric (training set)
        y_list_tain = data_train(:, end);
        y_min_values_tain = final_y_intervals_train(:, 1);
        y_max_values_tain = final_y_intervals_train(:, 2);
        index_V_site_train(i_site) = compute_V_metric(y_list_tain, y_min_values_tain, y_max_values_tain);
    end
    mean_index_V_site_train(i_run) = mean(index_V_site_train);
    
    time_gran_dpc(i_run) = toc;
    time_gran_dpc(i_run) = time_gran_dpc(i_run)+ time_dpc(i_run);
    
    % Evaluate granularity (test set)
    % 4. Justifiable granularity principle obtains interval-valued y corresponding to each prototype (test set)
    [U_global_R1_test] = step5_gran_test(test_data,m,n_cluster,n_D,center_global_R1);  % Obtain global membership matrix for test set
    
    % 5. Fuzzy inference model calculates revenue value intervals (test set)
    [results,final_y_intervals] = step4_new_gran_y(test_data,U_global_R1_test);  % Obtain interval-valued y for test set
    
    % 6. Calculate granularity metric (test set)
    y_list = test_data(:, end);
    y_min_values = final_y_intervals(:, 1);
    y_max_values = final_y_intervals(:, 2);
    index_V_test(i_run) = compute_V_metric(y_list, y_min_values, y_max_values);
end

%% Save process data
algName = 'dpc_gfrbm';
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
time_dpc_sorted = sort(roundn(time_dpc, -4))';
max_time_dpc = max(time_dpc_sorted);
min_time_dpc = min(time_dpc_sorted);
avg_time_dpc = mean(time_dpc_sorted);
std_time_dpc = std(time_dpc_sorted);
var_time_dpc = var(time_dpc_sorted);

% Add statistics to table
t_time_asoc = table(time_dpc_sorted, max_time_dpc, min_time_dpc, avg_time_dpc, std_time_dpc, var_time_dpc, info, datetime('now'));
writetable(t_time_asoc, path, 'Sheet', 3, 'WriteMode', 'append', 'WriteRowNames', false, 'WriteVariableNames', false);

% Calculate granulation runtime statistics
time_gran_dpc_sorted = sort(roundn(time_gran_dpc, -4))';
max_time_gran_dpc = max(time_gran_dpc_sorted);
min_time_gran_dpc = min(time_gran_dpc_sorted);
avg_time_gran_dpc = mean(time_gran_dpc_sorted);
std_time_gran_dpc = std(time_gran_dpc_sorted);
var_time_gran_dpc = var(time_gran_dpc_sorted);

% Add statistics to table
t_time_gran_dpc = table(time_gran_dpc_sorted, max_time_gran_dpc, min_time_gran_dpc, avg_time_gran_dpc, std_time_gran_dpc, var_time_gran_dpc, info, datetime('now'));
writetable(t_time_gran_dpc, path, 'Sheet', 4, 'WriteMode', 'append', 'WriteRowNames', false, 'WriteVariableNames', false);

disp('Performance metric statistics have been successfully saved to Excel');