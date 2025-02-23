package.path = package.path .. ";./mino/?.lua;./mino/matrix/?.lua;./mino/error_handling/?.lua;./mino/matrix/operations/?.lua;"
local mino = require('mino')
local matrix = mino.Matrix

-- a = matrix.new({ data = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}} })
-- a:T(#a:shape(), #a:shape() - 1):print({shape = true, data=true, strides=true})
a = matrix.new({ dims = {2, 2, 1, 1}, data = 1 })
b = matrix.new({ dims = {2, 1, 1, 4}, data = 1 })
c = a:matmul(b)
-- a:print({shape = true, data=false, strides=true})
-- b:print({shape = true,  data=false, strides=true})
c:print({shape = true,  strides=true})
c:backward(matrix.new({ dims = {2, 2, 1, 4}, data = 1 }))

a:print({grad=true, shape = true, data=false, strides=true})
b:print({grad=true, shape = true, data=false, strides=true})
-- x:print({shape = true, grad=true})



-- package.path = package.path .. ";./error_handling/?.lua;./activations/?.lua;./utils/?.lua;./loss/?.lua;"
-- local error_handling = require('error_handling')
-- local mino = require('mino')
-- local activations = require('activations')

-- x = mino.Matrix({ data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}} })
-- y = mino.Matrix({ data = {{0}, {1}, {1}, {0}} })

-- layer = mino.layers.Linear({dims = {2, 1}, data = 0.5})
-- epochs = 2
-- learning_rate = 0.01
-- for i = 1, epochs do
--     local output = layer:forward(x)
--     local loss = mino.loss.cross_entropy(output, y)
--     loss:backward()

--     for i = 1, layer.weights.dims[1] * layer.weights.dims[2] do
--         layer.weights.data[i] = layer.weights.data[i] - learning_rate * layer.weights.grad[i]
--         layer.weights.grad[i] = 0
--     end
--     for i = 1, layer.bias.dims[1] * layer.bias.dims[2] do
--         layer.bias.data[i] = layer.bias.data[i] - learning_rate * layer.bias.grad[i]
--         layer.bias.grad[i] = 0
--     end

-- end