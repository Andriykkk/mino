package.path = package.path .. ";./mino/activations/?.lua;./mino/matrix/?.lua;./mino/loss/?.lua;./mino/layers/?.lua;./mino/error_handling/?.lua;./mino/matrix/operations/?.lua;"

local matrix = require('matrix')
local mino = {}

local activations = {
    relu = require('relu'),
    softmax = require('softmax'),
}

local layers = {
    linear = require('linear'),
}

local loss = {
    cross_entropy = require('cross_entropy'),
}

mino.Matrix = matrix
mino.activations = activations
mino.layers = layers
mino.loss = loss

local parameters_mt = {
    __index = {
        add = function(self, item)
            self.parameters[self.size] = item
            self.size = self.size + 1
        end
    }
}

function mino.Parameters()
    local parameters = { parameters = {}, size = 0 }

    setmetatable(parameters, parameters_mt)

    return parameters
end



return mino
