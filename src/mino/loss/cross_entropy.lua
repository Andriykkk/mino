local cross_entropy = {}
local softmax = require('softmax')
local matrix = require('matrix')

local cross_entropy_mt = {
    __index = {
        new = cross_entropy.new
    },
    __call = function(self, input, target) return self.cross_entropy(input, target) end
}

setmetatable(cross_entropy, cross_entropy_mt)

function cross_entropy.cross_entropy(input, target)
    assert(input and input.dims, "Input must be a matrix with dimensions.")
    assert(target and target.dims, "Target must be a matrix with dimensions.")
    assert(input.dims[1] == target.dims[1], "Input and target batch sizes must match.")
    assert(#input.sub_dims == 0 and #target.sub_dims == 0, "Input and target must be 2d flat matrices.")

    local result = softmax(input)

    local batch_size = input.dims[1]
    local num_classes = input.dims[2]
    local loss = 0

    for i = 1, batch_size do
        local target_idx = target.data[i] + 1
        assert(target_idx >= 1 and target_idx <= num_classes, "Target index out of range.")
        local log_prob = math.log(result.data[(i - 1) * num_classes + target_idx])
        loss = loss - log_prob
    end

    loss = loss / batch_size

    local loss_tensor = matrix.new({dims = {1, 1}, data = loss})

    loss_tensor.backward = cross_entropy.cross_entropy_backward
    loss_tensor.operand1 = input
    loss_tensor.operand2 = target

    return loss_tensor
end

function cross_entropy.cross_entropy_backward(self, respect)
    assert(#self.operand1.sub_dims == 0, "Input must be a 2d flat matrix.")
    assert(#self.operand2.sub_dims ==  0, "Target must be a 2d flat matrix.")
    local input = self.operand1
    local target = self.operand2

    local probs = softmax(input)

    local grad_input = input:copy({data = 0})

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

function cross_entropy.new()
    return cross_entropy.cross_entropy
end

return cross_entropy