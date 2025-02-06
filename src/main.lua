package.path = package.path .. ";./error_handling/?.lua;./activations/?.lua;./utils/?.lua;./loss/?.lua;"
local error_handling = require('error_handling')
local mino = require('mino')
local activations = require('activations')

a = mino.Matrix({dims = {2, 2}, data = 1})
b = mino.Matrix({dims = {2, 2}, data = 1})

g = a + b
c = -g

local respect = mino.Matrix({dims = {2, 2}, data = 4})
c:backward(respect)

print("a")
mino.print_matrix_grad(a)
print("b")
mino.print_matrix_grad(b)
 
-- local a = mino.Matrix({ data = {1, 2, 3}})
-- local output = mino.activations.softmax(a)
-- local target = mino.Matrix({ data = {1, 0, 0}})
-- target = -target
-- mino.print_matrix(output)
