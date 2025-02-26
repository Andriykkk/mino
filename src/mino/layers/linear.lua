local matrix = require('matrix')
local linear = {}

local linear_mt = {
    __index = linear,
    __call = function(self, ...) return self:forward(...) end
}

function linear.new(params)
    local weights = matrix.new({dims = { params.input, params.output }, data = params.data})
    local bias = matrix.new({dims = { 1, params.output }, data = params.data})

    local layer = {weights = weights, bias = bias}

    setmetatable(layer, linear_mt)

    return layer
end

function linear:forward(input)
    local output = matrix.matmul(input, self.weights)
    output = output + self.bias
    return output
end

return linear