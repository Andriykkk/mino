package.path = package.path .. ";./mino/?.lua;./utils/datasets/?.lua;./utils/?.lua;"
local mino = require('mino')
local utils = require('utils')
local matrix = mino.Matrix

local a = matrix.new({ data = {{1, 2, 3}, {4, 5, 6}}, dims = {2, 3} })
local b = a:argmax()
b:print()