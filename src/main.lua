package.path = package.path .. ";./mino/?.lua;"
local mino = require('mino')
local matrix = mino.Matrix
local layers = mino.layers
local activations = mino.activations
local loss = mino.loss
local optimisers = mino.optimisers

a = matrix.new({ data = {{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}} })
target = matrix.new({ data = {{2}, {1}}})
b = loss.cross_entropy(a, target)
b:print({data = true, shape = true, strides = true})
b:sum():backward()
a:print({grad = true, data = true, shape = true, strides = true})

local function read_scv(file_path)
    local data = {}

    local file = io.open(file_path, "r")
    if not file then
        return nil
    end

    -- local index = 0
    for line in file:lines() do
        -- index = index + 1
        -- if index > 100 then
        --     break
        -- end
        local row = {}
        for value in string.gmatch(line, "([^,]+)") do
            table.insert(row, tonumber(value))
        end

        local label = row[1]

        local pixels = {}
        for i = 2, #row do
            table.insert(pixels, row[i]/255.0)
        end

        table.insert(data, {label = label, pixels = pixels})
    end

    file:close()

    return data
end

local function print_mnist(data, limit)
    for i = 1, limit do
        local row = data[i]
        local label = row.label
        local pixels = row.pixels

        print("Label: " .. label)

        for j = 1, 28 do
            for k = 1, 28 do
                local pixel = pixels[(j - 1) * 28 + k]
                if pixel > 0.7 then
                    io.write("#")
                elseif pixel > 0.3 then
                    io.write(".")
                else
                    io.write(" ")
                end
            end
            io.write("\n")
        end
    end
end

-- local dataset = read_scv('../mnist/mnist_train.csv')
-- print_mnist(dataset, 5)

-- -- mnist
-- local model = mino.Parameters()
-- model:add(layers.linear.new({ input = 28*28, output = 128}))
-- model:add(activations.relu.new())
-- model:add(layers.linear.new({ input = 128, output = 64}))
-- model:add(activations.relu.new())
-- model:add(layers.linear.new({ input = 64, output = 10}))

-- local criterion = loss.cross_entropy.new()
-- local optimiser = optimisers.sgd.new({ learning_rate = 0.01 })

-- local epoch = 5
-- for i = 1, epoch do
--     -- model.train()
--     for input, target in ipairs(trainset) do
--         optimiser.zero_grad(model)

--         output = model:forward(input)
--         loss = criterion:forward(output, target)
--         loss:backward()
--         optimiser.step(model)
--     end

--     print("Epoch " .. i .. " loss: " .. loss)
-- end

-- -- modal.eval()
-- total = 0
-- correct = 0
-- for input, target in ipairs(testset) do
--     output = model:forward(input)
--     predicted = torch.max(output, 1)
--     total = total + 1
--     if predicted[1] == target then
--         correct = correct + 1
--     end
-- end
-- print("Accuracy: ", correct / total)