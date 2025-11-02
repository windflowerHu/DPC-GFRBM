clc;
clear all;
close all;

matrix_data_name = {'1,airfoil_self_noise','2,parkinsons_updrs','3,compactiv','4,puma32h','5,fortune1000_2024'};


data_i = 1;% Dataset number, integer not exceeding 5
normway = 2;%Data normalization method, integer from 1-4
m = 2.0;%Fuzzy parameter
ratio = 0.3; % Data overlap ratio for each site
isOverlap = 0;  % Whether to allow data distribution overlap
run_time = 20; % Number of algorithm repetitions
P = 5; % Number of sites the data will be divided into
n_cluster = 2; % Number of clusters

model = 2;% set model category 1: DBSCAN-GFRBM,2: DF-GFRBM,3: FCM-GFRBM

 
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

        
% Initialize metrics
time_dbscan = zeros(run_time,1);  % Algorithm runtime
time_gran_dbscan= zeros(run_time,1);  % Granulation runtime
mean_index_V_site_train = zeros(run_time,1);  % raining set performance metric
index_V_test = zeros(run_time,1); % Testing set performance metric  

for i_run = 1:run_time
    tic;
    switch model
        case 1
            [client_prototypes, client_membership, granularity_results] = dbscan_outputs_gfrbm(splitdata, P);
        case 2
            [client_prototypes, client_membership, granularity_results] = df_outputs_gfrbm(splitdata, P, n_cluster);
        case 3
            [client_prototypes, client_membership, granularity_results] = fcm_outputs_gfrbm(splitdata, P,n_cluster);
        otherwise
            warning('Please select correct model number\n');
    end
     time_dbscan(i_run) = toc;

    tic;
    parfor i_site = 1:P
        data_train = splitdata{i_site};

        upper_bounds_train = granularity_results{i_site}(:, 7); 
        lower_bounds_train = granularity_results{i_site}(:, 8);
        bounds_matrix_train = [upper_bounds_train, lower_bounds_train];

        y_intervals_train = client_membership{i_site} * bounds_matrix_train; 
        y_values_train = [y_intervals_train(:, 2), y_intervals_train(:, 1)];

        y_list_tain = data_train(:, end);
        y_min_values_tain = y_values_train(:, 1);
        y_max_values_tain = y_values_train(:, 2); 
        index_V_site_train(i_site) = compute_V_metric(y_list_tain, y_min_values_tain, y_max_values_tain);
    end
    mean_index_V_site_train(i_run) = mean(index_V_site_train);

    time_gran_dbscan(i_run) = toc;
    time_gran_dbscan(i_run) = time_gran_dbscan(i_run)+ time_dbscan(i_run);
    
    % 初始化
    num_test_points = size(test_data, 1);
    y_intervals_clients = zeros(num_test_points, 2, P); % Third dimension is client index
    
    % Iterate through each client to calculate membership matrix and interval values
    for i_site = 1:P
        % Current client's clustering centers
        client_center = client_prototypes{i_site}; % Client's clustering centers
        
        % Calculate distance matrix from data points to client clustering centers
        tmp_data = test_data(:, 1:end-1); % Remove last column
        dist_data_center = pdist2(tmp_data, client_center);
        
        % Prevent division by zero
        dist_data_center(dist_data_center == 0) = 1e-6;
        
        % Fuzzification parameter
        uf = -2 / (m - 1);
        
        % Calculate membership matrix
        U_client_test = (dist_data_center .^ uf) ./ (sum(dist_data_center .^ uf, 2) * ones(1, size(client_center, 1)));
        
        % Extract interval value boundaries for current client
        upper_bounds_site = granularity_results{i_site}(:, 7); % Extract upper bounds
        lower_bounds_site = granularity_results{i_site}(:, 8); % Extract lower bounds
        bounds_matrix_site = [upper_bounds_site, lower_bounds_site];
        
        % Calculate interval values for this client on the test set
        y_intervals_site = U_client_test * bounds_matrix_site;
        
        % Store interval values for each client
        y_intervals_clients(:, :, i_site) = [y_intervals_site(:, 2), y_intervals_site(:, 1)]; % Each row is [min value, max value]
    end
    
    % Average interval values across all clients to get final interval values
    y_intervals_final = mean(y_intervals_clients, 3); % Average across client dimension
    y_values = [y_intervals_final(:, 1), y_intervals_final(:, 2)]; % Each row is [min value, max value]
    
    % Evaluate granularity (test set)
    y_list = test_data(:, end); % Ground truth values
    y_min_values = y_values(:, 1); % Minimum values
    y_max_values = y_values(:, 2); % Maximum values

    index_V_test(i_run) = compute_V_metric(y_list, y_min_values, y_max_values);


end

%% Save process data
switch model
    case 1
        algName = 'dbscan_gfrbm';
    case 2
        algName = 'df_gfrbm';
    case 3
        algName = 'fcm_gfrbm';
end

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
time_dbscan_sorted = sort(roundn(time_dbscan, -4))';
max_time_dbscan = max(time_dbscan_sorted);
min_time_dbscan = min(time_dbscan_sorted);
avg_time_dbscan = mean(time_dbscan_sorted);
std_time_dbscan = std(time_dbscan_sorted);
var_time_dbscan = var(time_dbscan_sorted);

% Add statistics to table
t_time_asoc = table(time_dbscan_sorted, max_time_dbscan, min_time_dbscan, avg_time_dbscan, std_time_dbscan, var_time_dbscan, info, datetime('now'));
writetable(t_time_asoc, path, 'Sheet', 3, 'WriteMode', 'append', 'WriteRowNames', false, 'WriteVariableNames', false);

% Calculate granulation runtime statistics
time_gran_dbscan_sorted = sort(roundn(time_gran_dbscan, -4))';
max_time_gran_dbscan = max(time_gran_dbscan_sorted);
min_time_gran_dbscan = min(time_gran_dbscan_sorted);
avg_time_gran_dbscan = mean(time_gran_dbscan_sorted);
std_time_gran_dbscan = std(time_gran_dbscan_sorted);
var_time_gran_dbscan = var(time_gran_dbscan_sorted);

% Add statistics to table
t_time_gran_dbscan = table(time_gran_dbscan_sorted, max_time_gran_dbscan, min_time_gran_dbscan, avg_time_gran_dbscan, std_time_gran_dbscan, var_time_gran_dbscan, info, datetime('now'));
writetable(t_time_gran_dbscan, path, 'Sheet', 4, 'WriteMode', 'append', 'WriteRowNames', false, 'WriteVariableNames', false);

disp('Performance metric statistics have been successfully saved to Excel');
