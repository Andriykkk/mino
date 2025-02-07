local cross_entropy = {}
local utils = require('utils')
local matrix = require('matrix')
local error_handling = require('error_handling')

-- function cross_entropy.cross_entropy_backward(input, target)
--     -- Check if target is one-hot encoded or a label vector
--     if target.dims ~= nil then
--         if input.dims[1] ~= target.dims[1] then
--             error_handling.show_error("Input and target dimensions are not compatible.")
--         end
--         target = target.data
--     else
--         if input.dims[1] ~= #target then
--             error_handling.show_error("Input and target dimensions are not compatible.")
--         end
--     end

--     -- Initialize the gradient matrix
--     local grad_input = matrix.new({dims = {input.dims[1], input.dims[2]}})

--     -- One-hot encode the target vector
--     local one_hot = utils.one_hot(target, input.dims[2])

--     -- Compute the gradient for each element in the matrix (logits)
--     for i = 1, input.dims[1] do
--         for j = 1, input.dims[2] do
--             -- Compute the gradient for the current element
--             grad_input.data[(i - 1) * input.dims[2] + j] = input.data[(i - 1) * input.dims[2] + j] - one_hot.data[(i - 1) * input.dims[2] + j]
--         end
--     end

--     return grad_input
-- end

function cross_entropy.cross_entropy(input, target)
    -- check for errors
    if target.dims ~= nil then
        if input.dims[1] ~= target.dims[1] then
            error_handling.show_error("Input and target dimensions are not compatible.")
        end
        target = target.data
    else
        if input.dims[1] ~= #target then
            error_handling.show_error("Input and target dimensions are not compatible.")
        end
    end

    local loss = matrix.new({dims = {input.dims[1], 1}})
    local that_small_value_that_prevent_crash = 1e-7

    local one_hot = utils.one_hot(target, input.dims[2])
    for i = 1, input.dims[1] do
        for j = 1, input.dims[2] do
            local pred = math.log(math.max(that_small_value_that_prevent_crash, input.data[(i - 1) * input.dims[2] + j]))
            loss.data[i] = loss.data[i] - one_hot.data[(i - 1) * input.dims[2] + j] * pred
        end
    end

    return loss
end

return cross_entropy