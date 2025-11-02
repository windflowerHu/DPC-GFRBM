function data = dataNormalization(data_ori, model)
switch model
    case 1
        data=(data_ori-repmat(min(data_ori),size(data_ori,1),1))./(repmat(max(data_ori),size(data_ori,1),1)-repmat(min(data_ori),size(data_ori,1),1)+eps);  
    case 2
        data=(data_ori-repmat(mean(data_ori),size(data_ori,1),1))./(repmat(sqrt(std(data_ori).^2+eps),size(data_ori,1),1));
 
    case 3
        data = data_ori./(repmat(sum(data_ori),size(data_ori,1),1)+eps);
    case 4
        data = data_ori;
    otherwise
        warning('error normalization.');
end