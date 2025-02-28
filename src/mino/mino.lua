package.path = package.path .. ";./mino/activations/?.lua;./mino/utils/?.lua;./mino/optimisers/?.lua;./mino/matrix/?.lua;./mino/loss/?.lua;./mino/layers/?.lua;./mino/error_handling/?.lua;./mino/matrix/operations/?.lua;"

local matrix = require('matrix')
local mino = {}

math.randomseed(os.time())

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

local optimisers = {
    sgd = require('sgd'),
}

mino.Matrix = matrix
mino.activations = activations
mino.layers = layers
mino.loss = loss
mino.optimisers = optimisers

local parameters_mt = {
    __index = function(self, key)
        if type(key) == "number" then
            return self.parameters[key]  
        else
            local methods = {
                add = function(self, item)
                    table.insert(self.parameters, item)
                end,
                train = function(self)
                    return
                end,
                eval = function(self)
                    return
                end
            }
            return methods[key]
        end
    end
}

function mino.Parameters()
    local table = { parameters = {} }

    setmetatable(table, parameters_mt)

    return table
end

function mino.Sequential(layers)
    local model = {layers = layers}

    function model:forward(input)
        for i = 1, #self.layers do
            input = self.layers[i](input)
        end
        return input
    end

    return model
end



return mino
