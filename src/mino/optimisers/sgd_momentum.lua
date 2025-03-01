local utils = require("mino.optimisers.utils")
local sdg_momentum = {}

local function apply_gradients(matrix, params)
    for i = 1, matrix.size do
        matrix.velocity[i] = params.momentum * matrix.velocity[i] + (1 - params.momentum) * matrix.grad[i]
        matrix.data[i] = matrix.data[i] - params.lr * matrix.velocity[i]
    end
end

function create_velocity(matrix)
    matrix.velocity = {}
    for i = 1, matrix.size do
        matrix.velocity[i] = 0
    end
end

function sdg_momentum.new(params)
    assert(params ~= nil, "Params cannot be nil in sgd")
    assert(params.parameters.parameters ~= nil, "Parameters cannot be nil in sgd")

    if params.learning_rate == nil then
        print("Learning rate not specified, defaulting to 0.01")
        params.learning_rate = 0.01
    end
    if params.momentum == nil then
        print("Momentum not specified, defaulting to 0.9")
        params.momentum = 0.9
    end

    local optimiser = { parameters = params.parameters.parameters, learning_rate = params.learning_rate, momentum = params.momentum }

    utils.parameters_worker(optimiser, create_velocity)

    function optimiser:step()
        utils.parameters_worker(optimiser, apply_gradients, {lr = self.learning_rate, momentum = self.momentum, optimiser = "sgd_momentum"})
    end

    optimiser.zero_grad = utils.zero_grad

    return optimiser
end

return sdg_momentum