local matrix = require('matrix')
local linear = {}

local linear_mt = {
    __index = linear,
    __call = function(self, ...) return self:forward(...) end
}

function linear.new(params)
    if params.initialisation == nil then
        params.initialisation = "xavier"
    end
    local weights = matrix.new({dims = { params.input, params.output }})
    local bias = matrix.new({dims = { 1, params.output }})

    if params.initialisation == "xavier" then
        matrix.initialise_matrix(weights, "xavier")
    end

    local layer = {weights = weights, bias = bias, parameters = {weights, bias}}

    setmetatable(layer, linear_mt)

    return layer
end

function linear:forward(input)
    local output = input:matmul(self.weights)
    output = output + self.bias
    return output
end

return linear