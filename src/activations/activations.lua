local activations = {}

local relu = require('relu')

function activations.RELU(input)
    relu.relu(input)
end

return activations