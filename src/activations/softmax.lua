local softmax = {}
local matrix = require('matrix')

function softmax.softmax_backward(self, respect)
    local opearand1 = self.operand1

    local probs = self.data

    local result = matrix.new({dims = {opearand1.dims[1], opearand1.dims[2]}, required_grad = false})
    for i = 1, self.dims[1] * self.dims[2] do
        local sum_grad = 0

        for j = 1, self.dims[1] * self.dims[2] do
            if j == i then
                sum_grad = sum_grad + respect.data[j] * probs[i] * (1 - probs[i])
            else
                sum_grad = sum_grad - respect.data[j] * probs[i] * probs[j]
            end
        end

        result.data[i] = sum_grad
    end

    if opearand1.required_grad == true then
        for i = 1, opearand1.dims[1] * opearand1.dims[2] do
            opearand1.grad[i] = opearand1.grad[i] + result.data[i]
        end
    end

    if opearand1.backward then
        opearand1:backward(result)
    end
end

function softmax.softmax(input)
    -- max logit
    local max_logit = -math.huge
    for i = 1, #input.data do
        if input.data[i] > max_logit then
            max_logit = input.data[i]
        end
    end

    -- compute exp
    local exp = matrix.new({dims = {input.dims[1], input.dims[2]}})
    local sum_exp = 0
    for i = 1, #input.data do
        exp.data[i] = math.exp(input.data[i] - max_logit)
        sum_exp = sum_exp + exp.data[i]
    end

    -- transform to probabilities
    for i = 1, #input.data do
        exp.data[i] = exp.data[i] / sum_exp
    end

    exp.backward = softmax.softmax_backward
    exp.operand1 = input

    return exp
end

return softmax 