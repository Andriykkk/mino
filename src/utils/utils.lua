local utils = {}
local matrix = require('matrix')
local error_handling = require('error_handling')

function utils.one_hot(labels, num_classes)
    if num_classes == nil then
        error_handling.show_error("Number of classes must be specified.")
    end
    
    local one_hot = matrix.new({dims = {labels.dims[1], num_classes}})

    for i = 1, labels.dims[1] do
        for j = 1, num_classes do
            if j - 1 == labels.data[i] then
                one_hot.data[(i - 1) * num_classes + j] = 1
            else
                one_hot.data[(i - 1) * num_classes + j] = 0
            end
        end
    end
    return one_hot
end

return utils