package.path = package.path .. ";./mino/activations/?.lua;"

local matrix = require('matrix')
local mino = {}

local activations = {
    relu = require('relu'),
}

mino.Matrix = matrix
mino.activations = activations




return mino
