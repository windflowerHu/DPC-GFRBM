function [best_a,best_b] = compute_PGJ_onedim(y_all,ym,ymin,ymax)
% 获取最佳的a和b值
a_candi = ymin:0.05:ym;
b_candi = ym:0.05:ymax;

result_a = zeros(length(a_candi),1);
M_a = sum(y_all<ym)+1e-6;
parfor i_a = 1:length(a_candi)
    cov_a = sum(a_candi(i_a)<=y_all&y_all<ym)/M_a;
    sp_a = 1-(abs(ym-a_candi(i_a))/(abs(ym-ymin)+1e-6));
    result_a(i_a) = cov_a*sp_a;
end
[~,idx] = max(result_a);
best_a = a_candi(idx);

result_b = zeros(length(b_candi),1);
M_b = sum(y_all>=ym)+1e-6;
parfor i_b = 1:length(b_candi)
    cov_b = sum(b_candi(i_b)>=y_all&y_all>=ym)/M_b;
    sp_b = 1-(abs(ym-b_candi(i_b))/(abs(ymax-ym)+1e-6));
    result_b(i_b) = cov_b*sp_b;
end
[~,idx] = max(result_b);
best_b = b_candi(idx);
