% function splitdata = split_data(data,P,ratio,isOverlap)
% n_data = size(data,1);
% y = data(:,end);
% splitdata = cell(P,1);
% if isOverlap
%     n_data_persite = floor(n_data*ratio);
%     for i = 1:P
%         tmp_indx = randi([1, n_data], n_data_persite, 1);
%         splitdata{i,1} = data(tmp_indx,:);
%     end
% else
%     n_data_persite = floor(n_data/P);
%     remainder = mod(n_data, P); % 余数
%     [~,ordy] = sort(y);
%     start_idx = 1;
%     for i = 1:P
%         if i <= remainder
%             end_idx = start_idx + n_data_persite;
%         else
%             end_idx = start_idx + n_data_persite -1;
%         end
%         splitdata{i} = data(ordy(start_idx:end_idx),:);
%         start_idx = end_idx + 1;
%     end
% 
% end

function splitdata = split_data(data, P, ratio, isOverlap)
n_data = size(data, 1);
y = data(:, end);
splitdata = cell(P, 1);

if isOverlap
    n_data_persite = floor(n_data * ratio);
    for i = 1:P
        tmp_indx = randi([1, n_data], n_data_persite, 1);
        splitdata{i} = data(tmp_indx, :);
    end
else
    n_data_persite = floor(n_data / P);
    remainder = mod(n_data, P);
    [~, ordy] = sort(y);
    start_idx = 1;

    for i = 1:P
        if i <= remainder
            end_idx = start_idx + n_data_persite;
        else
            end_idx = start_idx + n_data_persite - 1;
        end
        splitdata{i} = data(ordy(start_idx:end_idx), :);
        start_idx = end_idx + 1;
    end
end