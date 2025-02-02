local softmax = {}
local matrix = require('matrix')

function softmax.softmax(input)
    -- check errors
    local max_logit = 0
    for i = 1, #input.data do
        if input.data[i] < 0 then
            input.data[i] = 1e-7
        end
        if input.data[i] > 1 then
            input.data[i] = 1 - 1e-7
        end
        if input.data[i] > max_logit then
            max_logit = input.data[i]
        end
    end

    -- compute exp
    local exp = matrix.new({dims = {input.dims[1], input.dims[2]}})
    local sum_exp = 0
    for i = 1, #input.data do
        exp_val = math.exp(input.data[i] - max_logit)
        exp.data[i] = exp_val
        sum_exp = sum_exp + exp_val
    end

    -- transform to probabilities
    for i = 1, #input.data do
        exp.data[i] = input.data[i] / sum_exp
    end

    return exp
end

return softmax 