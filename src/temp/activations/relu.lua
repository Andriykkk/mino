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
    result.backward = relu.relu_backward
    result.operand1 = input
    return result
end

function relu.relu_backward(self, respect)
    for i = 1, #respect.data do
        if respect.data[i] < 0 then
            respect.data[i] = 0
        end
    end

    for i = 1, self.dims[1] * self.dims[2] do
        self.operand1.grad[i] = self.operand1.grad[i] + respect.data[i]
    end

    if self.operand1.backward then
        self:backward(respect)
    end
end

return relu