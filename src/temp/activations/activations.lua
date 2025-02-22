local activations = {}

local relu = require('relu')
local softmax = require('softmax')

function activations.relu(input)
    return relu.relu(input)
end

function activations.softmax(input)
    return softmax.softmax(input)
end

return activations