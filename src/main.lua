package.path = package.path .. ";./error_handling/?.lua;./activations/?.lua;"
local error_handling = require('error_handling')
local mino = require('mino')
local activations = require('activations')

 

local a = mino.Matrix({dims = {2, 3}, data = 0.1})
local b = mino.Matrix({dims = {4, 2}, data = 2})

local linear = mino.layers.Linear({dims = {3, 2}, data = 2})
local d = linear(a)
a = activations.SOFTMAX(a)
print("softmax")
mino.print_matrix(a)
