local activations = {}

local relu = require('relu')
local softmax = require('softmax')

function activations.RELU(input)
    return relu.relu(input)
end

function activations.SOFTMAX(input)
    return softmax.softmax(input)
end

return activations