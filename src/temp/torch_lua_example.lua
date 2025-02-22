-- Load necessary libraries
require 'torch'
require 'nn'
require 'optim'
require 'dataset-mnist'

-- Load MNIST dataset
local trainset = mnist.traindataset()
local testset = mnist.testdataset()

-- Preprocess data: flatten the images and normalize
local function preprocess(data)
    local inputs = data:clone():view(data:size(1), 28*28):double() / 255
    return inputs
end

trainset.data = preprocess(trainset.data)
testset.data = preprocess(testset.data)

-- Define a simple model with linear layers
local model = nn.Sequential()
model:add(nn.Linear(28*28, 128))  -- Input size (28x28) -> 128 neurons
model:add(nn.ReLU())              -- ReLU activation
model:add(nn.Linear(128, 64))     -- 128 -> 64 neurons
model:add(nn.ReLU())              -- ReLU activation
model:add(nn.Linear(64, 10))      -- 64 -> 10 output neurons (for 10 classes)
model:add(nn.LogSoftMax())        -- LogSoftMax for multi-class classification

-- Define loss function (Negative Log-Likelihood Loss)
local criterion = nn.ClassNLLCriterion()

-- Set up optimizer parameters
local optimState = {learningRate = 0.01}
local optimMethod = optim.sgd

-- Training loop
local function train(model, trainset)
    model:training()
    local epochLoss = 0
    for i = 1, trainset.size, 64 do
        local inputs = trainset.data:narrow(1, i, math.min(64, trainset.size - i + 1))
        local labels = trainset.labels:narrow(1, i, math.min(64, trainset.size - i + 1))
        
        local function feval(params)
            model:zeroGradParameters()
            local outputs = model:forward(inputs)
            local loss = criterion:forward(outputs, labels)
            local gradOutputs = criterion:backward(outputs, labels)
            model:backward(inputs, gradOutputs)
            return loss, model.gradParameters
        end
        
        local _, loss = optimMethod(feval, model.parameters, optimState)
        epochLoss = epochLoss + loss[1]
    end
    print("Training loss:", epochLoss / trainset.size)
end

-- Training for 5 epochs
for epoch = 1, 5 do
    print("Epoch", epoch)
    train(model, trainset)
end

-- Testing the model
local correct = 0
local total = 0
model:evaluate()
for i = 1, testset.size, 64 do
    local inputs = testset.data:narrow(1, i, math.min(64, testset.size - i + 1))
    local labels = testset.labels:narrow(1, i, math.min(64, testset.size - i + 1))
    local outputs = model:forward(inputs)
    local _, predicted = outputs:max(2)
    correct = correct + predicted:eq(labels):sum()
    total = total + labels:size(1)
end

print("Test Accuracy:", (correct / total) * 100)
