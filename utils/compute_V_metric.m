function index_V_new2 = compute_V_metric(ground_truth_y,y_L,y_R)
n_test_data = length(ground_truth_y);
covk_new2 = zeros(n_test_data,1);  
spk_new2 = zeros(n_test_data,1);  
range_y = max(ground_truth_y)-min(ground_truth_y);
parfor i_data = 1:n_test_data
    mk = min(abs(ground_truth_y(i_data)-y_L(i_data)),abs(ground_truth_y(i_data)-y_R(i_data)));
    nk = abs(y_R(i_data)-y_L(i_data))/2;
    if ground_truth_y(i_data)>=y_L(i_data) && ground_truth_y(i_data) <= y_R(i_data)
        covk_new2(i_data) = (1.0 + mk/nk)/2;
    else

        covk_new2(i_data) = (1.0 - mk/(nk + mk))/2;
    end

    spk_new2(i_data) = max(0, 1-(abs(y_L(i_data)-y_R(i_data))/range_y));

end
index_V_new2 = mean(covk_new2)*mean(spk_new2);

