package.path = package.path .. ";../matrix/?.lua"
local matrix = require('matrix')

local linear = {}

local linear_mt = {
    __index = linear,
    __call = function(self, ...) return self:forward(...) end
}

function linear.new(params, mino)
    local weights = mino.Matrix({dims = {params.dims[1], params.dims[2]}, data = params.data})
    local bias = mino.Matrix({dims = {1, params.dims[2]}, data = params.data})

    local layer = {weights = weights, bias = bias}

    setmetatable(layer, linear_mt)

    return layer
end

function linear:forward(input)
    local output = matrix.matmul(input, self.weights) + self.bias
    return output
end

return linear

