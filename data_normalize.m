function data=data_normalize(data,method)
    if strcmp(method,'range')
        data=(data-repmat(min(data),size(data,1),1))./(repmat(max(data),size(data,1),1)-repmat(min(data),size(data,1),1)+eps);
    elseif strcmp(method,'var')
        data=(data-repmat(mean(data),size(data,1),1))./(repmat(sqrt(std(data).^2+eps),size(data,1),1));
    else
        data = data./(repmat(sum(data),size(data,1),1)+eps);
    end
