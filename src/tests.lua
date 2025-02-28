package.path = package.path .. ";./mino/?.lua;"

local mino = require('mino')
local utils = require('utils')
local matrix = mino.Matrix
local layers = mino.layers
local activations = mino.activations
local mino_loss = mino.loss
local optimisers = mino.optimisers

function runTest(testName, testFunc)
    local status, err = pcall(testFunc)
    if not status then
        print("Test failed: " .. testName .. " Error: " .. err)
    else
        print("Test passed: " .. testName)
    end
end

function check_number_threshold(number, target, threshold)
    local status = math.abs(number - target) < threshold

    if not status then
        assert(false)
    end
end

function test1()
    local target_number = {2.30, 2.2, 2.11, 2.01, 1.93}

    local tensor = matrix.new({ data = 0.01, dims = {784, 10} })
    local bias = matrix.new({ data = 0.01, dims = {1,10} })
    local input = matrix.new({ data = 0.01, dims = {1, 784} })
    local target = matrix.new({ data = 5, dims = {1, 1} })
    local criterion = mino_loss.cross_entropy.new()
    local model = mino.Parameters()
    model:add(tensor)
    model:add(bias)
    local optimiser = optimisers.sgd.new({ parameters = model, learning_rate = 0.1 })

    for i = 1, 5 do
        result = input:matmul(tensor) + bias
        loss = criterion(result, target)
        loss:backward()

        optimiser:step()
        optimiser:zero_grad()
    end
end

function test2()
    local target_number = {2.30, 2.2, 2.11, 2.01, 1.93}
    
    local tensor = matrix.new({ data = 0.01, dims = {784, 10} })
    local bias = matrix.new({ data = 0.01, dims = {1,10} })
    local input = matrix.new({ data = 0.01, dims = {2, 784} })
    local target = matrix.new({ data = {{5} , {5}}, dims = {2, 1} })
    local criterion = mino_loss.cross_entropy.new()
    local model = mino.Parameters()
    model:add(tensor)
    model:add(bias)
    local optimiser = optimisers.sgd.new({ parameters = model, learning_rate = 0.1 })


    for i = 1, 5 do
        result = input:matmul(tensor) + bias
        loss = criterion(result, target)
        loss:backward()

        optimiser:step()
        optimiser:zero_grad()
        
        check_number_threshold(loss.data[1], target_number[i], 0.02)
    end
end

function test3()
    local target_number = {2.30, 2.26, 2.22, 2.18, 2.14}
    
    local tensor = matrix.new({ data = 0.01, dims = {784, 10} })
    local bias = matrix.new({ data = 0.01, dims = {1,10} })
    local input = matrix.new({ data = 0.01, dims = {2, 784} })
    local target = matrix.new({ data = {{5} , {7}}, dims = {2, 1} })
    local criterion = mino_loss.cross_entropy.new()
    local model = mino.Parameters()
    model:add(tensor)
    model:add(bias)
    local optimiser = optimisers.sgd.new({ parameters = model, learning_rate = 0.1 })


    for i = 1, 5 do
        result = input:matmul(tensor) + bias
        loss = criterion(result, target)
        loss:backward()

        optimiser:step()
        optimiser:zero_grad()
        
        check_number_threshold(loss.data[1], target_number[i], 0.02)
    end
end

function test4()
    local target_number = {2.30, 2.2, 2.11, 2.01, 1.93}

    local tensor = matrix.new({ data = 0.01, dims = {784, 10} })
    local bias = matrix.new({ data = 0.01, dims = {1,10} })
    local input = matrix.new({ data = 0.01, dims = {2, 784} })
    local target = matrix.new({ data = {{5} , {5}}, dims = {2, 1} })
    local criterion = loss.cross_entropy.new()
    local model = mino.Parameters()
    model:add(tensor)
    model:add(bias)
    local relu = activations.relu.new()
    local optimiser = optimisers.sgd.new({ parameters = model, learning_rate = 0.1 })


    for i = 1, 5 do
        result = input:matmul(tensor) + bias
        result = relu(result)
        loss = criterion(result, target)
        loss:backward()

        optimiser:step()
        optimiser:zero_grad()
        
        loss:print()
    end
end

runTest("test1: crossentropy with weight and bias matrix", test1)
runTest("test2: crossentropy with weight, bias and batches", test2)
runTest("test3: crossentropy with weight, bias and batches", test3)
runTest("test4: previous test with relu", test3)