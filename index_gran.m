function index_V_new2 = index_gran(ground_truth_y,y_L,y_R)
n_test_data = length(ground_truth_y);
covk_new2 = zeros(n_test_data,1);  % 存储覆盖率的数组
spk_new2 = zeros(n_test_data,1);  % 存储特异性的数组
range_y = max(ground_truth_y)-min(ground_truth_y);  % 真实值的范围 range_y
parfor i_data = 1:n_test_data
    mk = min(abs(ground_truth_y(i_data)-y_L(i_data)),abs(ground_truth_y(i_data)-y_R(i_data)));
    nk = abs(y_R(i_data)-y_L(i_data))/2;
    if ground_truth_y(i_data)>=y_L(i_data) && ground_truth_y(i_data) <= y_R(i_data)
        % covk_new2(i_data) = 1.0;  % 如果真实值在左右边界之间，则将覆盖率设置为1.0   
        covk_new2(i_data) = (1.0 + mk/nk)/2;
    else
        % covk_new2(i_data) = (1.0-(mk/(nk+mk)))/2;
        % covk_new2(i_data) = 0.0;
        covk_new2(i_data) = (1.0 - mk/(nk + mk))/2;
    end
    % spk_new2(i_data) = exp(y_L(i_data)-y_R(i_data));
    spk_new2(i_data) = max(0, 1-(abs(y_L(i_data)-y_R(i_data))/range_y));  % 特异性：真实值与左右边界的差异除以真实值范围
    % spk_new2(i_data) = 1 / (1 + abs(y_L(i_data) - y_R(i_data)) / range_y); % 使用非线性惩罚

end
index_V_new2 = mean(covk_new2)*mean(spk_new2);

% disp(['粒度的性能评价指标为: ', num2str(index_V_new2)]);
