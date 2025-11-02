function [center, U, t, obj_fcn] = FCM(data, n_clusters, m, iterMax, thred, center_ini)
    d = size(data, 2);
    center = center_ini;
    obj_fcn = zeros(iterMax, 1);

    for t = 1:iterMax
        dist_data_center = pdist2(data, center);
        dist_data_center(dist_data_center == 0) = 0.000001;% avoid invalid value
        
        uf = -2 / (m - 1);
        u = (dist_data_center.^uf) ./ (sum(dist_data_center.^uf, 2) * ones(1, n_clusters));
        mf = u.^m;
        obj_fcn(t) = sum(sum((dist_data_center.^2) .* mf)); 
        center = ((u.^m)' * data) ./ (((sum(u.^m, 1))') * ones(1, d));
        
        if t > 1
            if abs(obj_fcn(t) - obj_fcn(t-1)) < thred
                break;
            end
        end
    end

    U = u;
    obj_fcn(t+1:iterMax) = [];
end