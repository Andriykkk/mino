local relu = {}
local matrix = require('matrix')

function relu.relu(input)
    local result = matrix.new({dims = {input.dims[1], input.dims[2]}})
    for i = 1, #input.data do
        if input.data[i] < 0 then
            result.data[i] = 0
        else
            result.data[i] = input.data[i]
        end
    end 
    return result
end

return relu