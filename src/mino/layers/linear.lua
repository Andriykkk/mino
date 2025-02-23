local matrix = require('matrix')
local linear = {}

local linear_mt = {
    __index = linear,
    __call = function(self, ...) return self:forward(...) end
}

function linear.new(params)
    local weights = matrix.Matrix({dims = params.dims, data = params.data})
    params.dims[#params.dims - 1] = 1
    local bias = matrix.Matrix({dims = params.dims, data = params.data})

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