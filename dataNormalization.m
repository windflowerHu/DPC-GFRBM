% function data_ori = dataNormalization(data_ori,model)
% switch model
%     case 1
%         data_ori=data_normalize(data_ori,'range');
%         data_ori=normr(data_ori);
%     case 2
%         data_ori=data_normalize(data_ori,'var');
%         data_ori=normr(data_ori);
%     case 3
%         data_ori=data_normalize(data_ori,'xy');
%         data_ori=normr(data_ori);
%     case 4
%         data_ori=normr(data_ori);
%     case 5
%         data_ori=data_normalize(data_ori,'range');
%     case 6
%         data_ori=data_normalize(data_ori,'var');
%     case 7
%         data_ori=data_normalize(data_ori,'xy');
%     case 8
%     otherwise
%         warning('error normalization.');
% end

function data_ori = dataNormalization(data_ori, model)
switch model
    case 1
        data_ori(:,1) = data_normalize(data_ori(:,1), 'range');
        data_ori(:,2) = data_normalize(data_ori(:,2), 'range');
        data_ori(:,3) = data_normalize(data_ori(:,3), 'range');
        data_ori = normr(data_ori);
    case 2
        data_ori(:,1) = data_normalize(data_ori(:,1), 'var');
        data_ori(:,2) = data_normalize(data_ori(:,2), 'var');
        data_ori(:,3) = data_normalize(data_ori(:,3), 'var');
        data_ori = normr(data_ori);
    case 3
        data_ori(:,1) = data_normalize(data_ori(:,1), 'xy');
        data_ori(:,2) = data_normalize(data_ori(:,2), 'xy');
        data_ori(:,3) = data_normalize(data_ori(:,3), 'xy');
        data_ori = normr(data_ori);
    case 4
        data_ori = normr(data_ori);
    case 5
        data_ori(:,1) = data_normalize(data_ori(:,1), 'range');
        data_ori(:,2) = data_normalize(data_ori(:,2), 'range');
        data_ori(:,3) = data_normalize(data_ori(:,3), 'range');
    case 6
        data_ori(:,1) = data_normalize(data_ori(:,1), 'var');
        data_ori(:,2) = data_normalize(data_ori(:,2), 'var');
        data_ori(:,3) = data_normalize(data_ori(:,3), 'var');
    case 7
        data_ori(:,1) = data_normalize(data_ori(:,1), 'xy');
        data_ori(:,2) = data_normalize(data_ori(:,2), 'xy');
        data_ori(:,3) = data_normalize(data_ori(:,3), 'xy');
    case 8
    otherwise
        warning('error normalization.');
end