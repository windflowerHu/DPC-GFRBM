function [center_site_local_R1, U_site_local_R1] = ...
    step1_local_fcm(splitdata,P,m,n_cluster)

n_split_data = zeros(P,1);
for i_site = 1:P
    n_split_data(i_site) = size(splitdata{i_site},1);
end

%% 每端的聚类中心和划分矩阵
center_site_local_R1 = cell(P,1);
U_site_local_R1 = cell(P,1);
for i_site = 1:P
    tmp_data = splitdata{i_site};
    tmp_data = tmp_data(:, 1:end-1); % 去掉最后一列
    tmp_n_data = size(tmp_data,1);
    tmp_indx = randperm(tmp_n_data, n_cluster);
    center_ini = tmp_data(tmp_indx,:);
    [center_site_local_R1{i_site},U_site_local_R1{i_site}] = FCM(tmp_data, n_cluster, m, 200, 1e-5, center_ini);
    U_site_local_R1{i_site} = U_site_local_R1{i_site}';
end