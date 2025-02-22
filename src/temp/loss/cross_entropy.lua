local cross_entropy = {}
local utils = require('utils')
local matrix = require('matrix')
local error_handling = require('error_handling')
local softmax = require('activations').softmax

function cross_entropy.cross_entropy_backward(self)
    local input = self.operand1
    local target = self.operand2

    local probs = softmax(input)

    local grad_input = matrix.new({dims = {input.dims[1], input.dims[2]}, required_grad = false})

    for i = 1, input.dims[1] do
        for j = 1, input.dims[2] do
            grad_input.data[(i - 1) * input.dims[2] + j] = (probs.data[(i - 1) * input.dims[2] + j] - (j - 1 == target[i] and 1 or 0)) / input.dims[1]
        end
    end

    if input.required_grad == true then
        for i = 1, input.dims[1] * input.dims[2] do
            input.grad[i] = input.grad[i] + grad_input.data[i]
        end
    end

    if input.backward then
        input:backward(grad_input)
    end
end


function cross_entropy.cross_entropy(input, target)
    -- check for errors
    if target.dims ~= nil then
        if input.dims[1] ~= target.dims[1] then
            error_handling.show_error("Input and target dimensions are not compatible.")
        end
    else
        if input.dims[1] ~= #target then
            error_handling.show_error("Input and target dimensions are not compatible.")
        end
    end

    local probs = softmax(input)
    
    local loss = 0
    for i = 1, input.dims[1] do
        for j = 1, input.dims[2] do
            if j == target[i] then
                loss = loss - math.log(math.max(1e-7, probs.data[(i - 1) * input.dims[2] + j]))
            end
        end
    end

    loss = loss / input.dims[1]

    local loss_tensor = matrix.new({dims = {1, 1}, data = loss})
    loss_tensor.backward = cross_entropy.cross_entropy_backward
    loss_tensor.operand1 = input
    loss_tensor.operand2 = target

    return loss_tensor
end

return cross_entropy