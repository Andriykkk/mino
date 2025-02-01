package.path = package.path .. ";./error_handling/?.lua;./activations/?.lua;"
local error_handling = require('error_handling')
local mino = require('mino')
local activations = require('activations')

 

local a = mino.Matrix({dims = {2, 3}, data = 2})
local b = mino.Matrix({dims = {4, 2}, data = 2})

local linear = mino.layers.Linear({dims = {3, 2}, data = 2})
local d = linear(a)
activations.RELU(d)
mino.print_matrix(linear.weights)
print("bias")
mino.print_matrix(linear.bias)
mino.print_matrix(d)
