% function [center,U,t,obj_fcn] = FCM(data, n_clusters, m, iterMax, thred, center_ini)
% d = size(data,2);
% center = center_ini;
% obj_fcn = zeros(iterMax, 1);
% 
% % figure(12)
% % scatter(data(:,1),data(:,2),'.','k');
% % hold on
% % scatter(center(:,1),center(:,2),'o','y');
% % hold on
% 
% 
% for t = 1:iterMax
%     dist_data_center = pdist2(data,center);
%     dist_data_center(dist_data_center==0) = 0.000001;
% 
%     uf = -2/(m-1);
%     u = (dist_data_center.^uf)./ (sum(dist_data_center.^uf,2)*ones(1,n_clusters));
%     mf = u.^m;
%     obj_fcn(t) = sum(sum((dist_data_center.^2).*mf));
%     center = ((u.^m)'*data)./(((sum(u.^m,1))')*ones(1,d));
% 
% 
% %     center = ((u.^m)*data+w*G)./(((sum(u.^m,1))'+lamda*sum(w,2))*ones(1,d));
% 
% %     scatter(center(1,1),center(1,2),'o','g');
% %     hold on
% %     scatter(center(2,1),center(2,2),'o','r');
% %     hold on
% %     scatter(center(3,1),center(3,2),'o','b');
% 
%     if t > 1
% %         if max(u-u_old)<thred
%         if abs(obj_fcn(t)-obj_fcn(t-1))<thred
%             break;
% %         else
% %             u_old = u;
%         end
% %     else
% %         u_old = u;
%     end
% end
% 
% U = u;
% obj_fcn(t+1:iterMax) = [];
% t;


function [center, U, t, obj_fcn] = FCM(data, n_clusters, m, iterMax, thred, center_ini)
    d = size(data, 2);
    center = center_ini;
    obj_fcn = zeros(iterMax, 1); % 初始化目标函数值数组

    for t = 1:iterMax
        dist_data_center = pdist2(data, center);
        dist_data_center(dist_data_center == 0) = 0.000001; % 避免除以零
        
        uf = -2 / (m - 1);
        u = (dist_data_center.^uf) ./ (sum(dist_data_center.^uf, 2) * ones(1, n_clusters));
        mf = u.^m;
        obj_fcn(t) = sum(sum((dist_data_center.^2) .* mf)); % 计算当前迭代的目标函数值
        center = ((u.^m)' * data) ./ (((sum(u.^m, 1))') * ones(1, d));
        
        if t > 1
            if abs(obj_fcn(t) - obj_fcn(t-1)) < thred
                break;
            end
        end
    end

    U = u;
    obj_fcn(t+1:iterMax) = []; % 删除未使用的预分配空间
    
    %% 绘制收敛曲线
    figure;
    plot(1:t, obj_fcn, '-');
    xlabel('Iterations');
    ylabel('Objective Function Value');
    % title('Convergence of Objective Function during FCM Clustering');
    grid on;
end