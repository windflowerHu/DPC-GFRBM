function [V_site_center,cell_A_opt, cell_Y_hat, rmse, mean_rmse, center_site] =cfc_gfrbm_prototype_train(splitdata,P,n_D,n_cluster,belta,sigma,n_data)
V = cell(P,1);
U = cell(P,1);
data_site_center = [];
for i_site = 1:P
    tmp_data = splitdata{i_site};
    tmp_indx = randperm(size(tmp_data,1), n_cluster);
    center_ini = tmp_data(tmp_indx,:);
    [V{i_site},tmp_U] = FCM(splitdata{i_site}, n_cluster, 2.0, 150, 1e-5, center_ini);
    U{i_site} = tmp_U';
    data_site_center = [data_site_center;V{i_site}];
end

tmp_indx = randperm(size(data_site_center,1), n_cluster);
center_ini = data_site_center(tmp_indx,:);
V_site_center = FCM(data_site_center, n_cluster, 2.0, 150, 1e-5, center_ini);  % 从本地FCM的聚类中心继续FCM，得到全局聚类中心

center_site = cell(P,1);
U_site = cell(P,1);
for i_site = 1:P
    tmp_data = splitdata{i_site};
    tmp_n_data = size(tmp_data,1);
    tmp_indx = randperm(tmp_n_data, n_cluster);
    center_ini = tmp_data(tmp_indx,:);
    [center_site{i_site},U_site{i_site}] = site_fcm(tmp_data,n_cluster, center_ini, V_site_center, belta);
end


art_data = cell(P,1);
n_per_art = floor(n_data/P/n_cluster);
U_art = cell(P,1);
uf = -2;
parfor i_site = 1:P
    tmp_center = center_site{i_site};
    for i_cluster = 1:n_cluster
       tmp_data= zeros(n_per_art,n_D);
       for i_D = 1:n_D
           tmp_data(:,i_D) = normrnd(tmp_center(i_cluster,i_D), sigma, [n_per_art, 1]);
       end
       art_data{i_site} = [art_data{i_site};tmp_data];
       dist_data_center = pdist2(tmp_data,tmp_center);
       tmp_U = (dist_data_center.^uf)./ (sum(dist_data_center.^uf,2)*ones(1,n_cluster));
       U_art{i_site} = [U_art{i_site};tmp_U];
    end
end

all_data = cell(P,1);
all_U = cell(P,1);
for i_site = 1:P
    all_data{i_site} = [splitdata{i_site};art_data{i_site}];
    all_U{i_site} = [(U{i_site})';U_art{i_site}];
end

[cell_A_opt, cell_Y_hat, rmse,mean_rmse] = site_model_train(all_data,P,all_U,n_cluster,n_D);
