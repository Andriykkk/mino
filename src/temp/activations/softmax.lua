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
    local probs = matrix.new({dims = {input.dims[1], input.dims[2]}})
    for i = 1, input.dims[1] do
        local max_logit = -math.huge
        for j = 1, input.dims[2] do
            if input.data[(i - 1) * input.dims[2] + j] > max_logit then
                max_logit = input.data[(i - 1) * input.dims[2] + j]
            end
        end

        local sum_exp = 0
        for j = 1, input.dims[2] do
            probs.data[(i - 1) * input.dims[2] + j] = math.exp(input.data[(i - 1) * input.dims[2] + j] - max_logit)
            sum_exp = sum_exp + probs.data[(i - 1) * input.dims[2] + j]
        end

        for j = 1, input.dims[2] do
            probs.data[(i - 1) * input.dims[2] + j] = probs.data[(i - 1) * input.dims[2] + j] / sum_exp
        end
    end

    probs.backward = softmax.softmax_backward
    probs.operand1 = input

    return probs
end

return softmax 