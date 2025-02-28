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
        result.data[i] = math.max(0, input.data[i])
    end
    
    result.backward = relu.relu_backward
    result.operand1 = input
    
    return result
end

function relu.relu_backward(self, respect)
    local input = self.operand1
    local result = input:copy({data = 0})

    for i = 1, input.size do
        if input.data[i] > 0 then
            result.data[i] = respect.data[i]
        else
            result.data[i] = 0
        end
    end

    if input.required_grad == true then
        for i = 1, input.size do
            input.grad[i] =  input.grad[i] + result.data[i]
        end
    end

    if input.backward then
        input:backward(result)
    end
end

function relu.new()
    return relu.relu
end

return relu