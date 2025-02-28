package.path = package.path .. ";./mino/?.lua;./utils/datasets/?.lua;./utils/?.lua;"
local mino = require('mino')
local utils = require('utils')
local matrix = mino.Matrix
local layers = mino.layers
local activations = mino.activations
local loss = mino.loss
local optimisers = mino.optimisers
local mnist_dataset = require('mnist_dataset')
local progress_bar = require('progress_bar')

-- local tensor = matrix.new({ data = 0.01, dims = {784, 10} })
-- local bias = matrix.new({ data = 0.01, dims = {1,10} })
-- local input = matrix.new({ data = 0.01, dims = {2, 784} })
-- local target = matrix.new({ data = {{5} , {5}}, dims = {2, 1} })
-- local criterion = loss.cross_entropy.new()
-- local model = mino.Parameters()
-- model:add(tensor)
-- model:add(bias)
-- local relu = activations.relu.new()
-- local optimiser = optimisers.sgd.new({ parameters = model, learning_rate = 0.1 })


-- for i = 1, 5 do
--     result = input:matmul(tensor) + bias
--     result = relu(result)
--     loss = criterion(result, target)
--     loss:backward()

--     optimiser:step()
--     optimiser:zero_grad()
    
--     loss:print()
-- end


local function get_batches(data, batch_size)
    local batches = {}

    for i = 1, #data, batch_size do
        local input = {}
        local target = {}
        for j = i, math.min(i + batch_size - 1, #data) do
            local row = data[j]
            local label = row.label
            local pixels = row.pixels

            table.insert(input, pixels)
            table.insert(target, {label})
        end
        table.insert(batches, {input, target})
    end

    return batches
end

local dataset = mnist_dataset.read_csv('../mnist/mnist_train.csv')
local training_set = get_batches(dataset, 64)

local testing_dataset = mnist_dataset.read_csv('../mnist/mnist_test.csv')
local testing_set = get_batches(testing_dataset, 64)

-- mnist
local model = mino.Parameters()
model:add(layers.linear.new({ input = 28*28, output = 128}))
model:add(activations.relu.new())
model:add(layers.linear.new({ input = 128, output = 64}))
model:add(activations.relu.new())
model:add(layers.linear.new({ input = 64, output = 10}))

local sequential = mino.Sequential(model.parameters)

local criterion = loss.cross_entropy.new()
local optimiser = optimisers.sgd.new({ parameters = model, learning_rate = 0.1 })

local bar = progress_bar:new(#training_set)

local epoch = 5
for i = 1, epoch do
    model.train()
    for i = 1, #training_set do
        local input = training_set[i][1]
        local target = training_set[i][2]

        optimiser.zero_grad(model)

        local input_matrix = matrix.new({ data = input })
        local target_matrix = matrix.new({ data = target })
        
        output = sequential:forward(input_matrix)
        loss = criterion(output, target_matrix)
        loss:backward()
        optimiser:step()
        bar:step({ loss = loss.data[1]})
        
    end
end

-- print("Testing")
-- model.eval()
-- total = 0
-- correct = 0
-- for i = 1, #testing_set do
--     local input = testing_set[i][1]
--     local target = testing_set[i][2]

--     local input_matrix = matrix.new({ data = input })
--     local target_matrix = matrix.new({ data = target })

--     output = sequential:forward(input_matrix)
--     predicted = output:argmax()
--     total = total + 1
--     print(predicted.data[1], target_matrix.data[1])
--     if predicted.data[1] == target_matrix.data[1] then
--         correct = correct + 1
--     end
-- end
-- print("Accuracy: ", correct / total)