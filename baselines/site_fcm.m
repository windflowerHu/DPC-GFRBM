function [center,U] = site_fcm(data,n_cluster, center_ini, V_site, belta)
%% Use global cluster centers to correct cluster centers at each site
iterMax = 150;
thred = 1e-5;
[n_data,d] = size(data);
center = center_ini;
obj_fcn = zeros(iterMax, 1);

for t = 1:iterMax
    dist_data_center = pdist2(data,center);
    dist_data_center(dist_data_center==0) = 0.000001;
    dist_V_center = pdist2(V_site,center);
    dist_V_center(dist_V_center==0) = 0.000001;
    dist_V_center = diag(dist_V_center);
    dist_V_center = dist_V_center';
    
    tmp1 = dist_data_center.^2;
    tmp2 = dist_V_center.^2;
    up = tmp1+belta*(ones(n_data,1)*tmp2);
    U = up./ (sum(up,2)*ones(1,n_cluster));
    mf = U.^2;
    obj_fcn(t) = sum(sum((dist_data_center.^2).*mf+belta*(ones(n_data,1)*(dist_V_center.^2)).*mf));
    
    center = ((mf)'*data+belta*(sum(mf)'*ones(1,d)).*V_site)./((sum(mf))'*ones(1,d)*(1+belta));
    
    
    if t > 1
        if abs(obj_fcn(t)-obj_fcn(t-1))<thred
            break;
        end
    end
end