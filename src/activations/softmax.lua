local softmax = {}
local matrix = require('matrix')

function softmax.softmax(input)
    -- max logit
    local max_logit = 0
    for i = 1, #input.data do
        if input.data[i] > max_logit then
            max_logit = input.data[i]
        end
    end

    local that_small_value_that_prevent_crash = 1e-7

    -- compute exp
    local exp = matrix.new({dims = {input.dims[1], input.dims[2]}})
    local sum_exp = 0
    for i = 1, #input.data do
        local exp_val = math.exp(math.max(that_small_value_that_prevent_crash, math.min(1 - that_small_value_that_prevent_crash, input.data[i] - max_logit)))
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