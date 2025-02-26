local matrix = require('matrix')
local relu = {}

local relu_mt = {
    __index = {
        new = relu.new
    },
    __call = function(self, input)
        return self.relu(input)
    end
}

setmetatable(relu, relu_mt)

function relu.relu(input)
    local result = input:copy({data = 0})

    for i = 1, input.size do
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
    for i = 1, respect.size do
        if respect.data[i] < 0 then
            respect.data[i] = 0
        end
    end

    if self.operand1.required_grad == true then
        for i = 1, respect.size do
            self.operand1.grad[i] = self.operand1.grad[i] + respect.data[i]
        end
    end

    if self.operand1.backward then
        self:backward(respect)
    end
end

function relu.new()
    return relu.relu
end

return relu