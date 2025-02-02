package.path = package.path .. ";./error_handling/?.lua;./activations/?.lua;./utils/?.lua;./loss/?.lua;"
local error_handling = require('error_handling')
local mino = require('mino')
local activations = require('activations')

 
local table = {1, 3, 2}
a = mino.Matrix({dims = {3, 2}, data = 0.5})
local b = mino.loss.cross_entropy(a, table)
mino.print_matrix(b)
