package.path = package.path .. ";./mino/?.lua;"
local mino = require('mino')
local matrix = mino.Matrix
local layers = mino.layers
local activations = mino.activations
local loss = mino.loss
local optimisers = mino.optimisers

a = matrix.new({ data = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}} })
b = activations.softmax.softmax(a)
b:print({data = true, shape = true, strides = true})
b:sum():backward()
a:print({grad = true, data = true, shape = true, strides = true})
-- a = matrix.new({ data = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}} })
-- a:T(#a:shape(), #a:shape() - 1):print({shape = true, data=true, strides=true})
-- a = matrix.new({ dims = {2, 2, 1, 1}, data = 1 })
-- b = matrix.new({ dims = {2, 2, 1, 4}, data = 1 })
-- c = a:matmul(b)
-- -- a:print({shape = true, data=false, strides=true})
-- -- b:print({shape = true,  data=false, strides=true})
-- c:print({shape = true,  strides=true})
-- -- c:backward(matrix.new({ dims = {2, 2, 1, 4}, data = 1 }))
-- c:sum():backward()

-- a:print({grad=true, shape = true, data=false, strides=true})
-- b:print({grad=true, shape = true, data=false, strides=true})


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