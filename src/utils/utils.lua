local utils = {}
local matrix = require('matrix')

function utils.one_hot(labels, num_classes)
    local one_hot = matrix.new({dims = {#labels, num_classes}})
    for i = 1, #labels do
        for j = 1, num_classes do
            if j == labels[i] then
                one_hot.data[(i - 1) * num_classes + j] = 1
            else
                one_hot.data[(i - 1) * num_classes + j] = 0
            end
        end
    end
    return one_hot
end

return utils