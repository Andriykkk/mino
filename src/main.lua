package.path = package.path .. ";./error_handling/?.lua;./activations/?.lua;./utils/?.lua;./loss/?.lua;"
local error_handling = require('error_handling')
local mino = require('mino')
local activations = require('activations')

 
local a = mino.Matrix({dims = {2, 2}, data = 1})
local b = mino.Matrix({dims = {2, 2}, data = 2})
local d = mino.Matrix({dims = {2, 2}, data = 3})
local respect = mino.Matrix({dims = {2, 2}, data = 4})
local h = d - b
local g = d + h
local c = mino.matmul(a, g)

c:backward(respect)

print("a")
mino.print_matrix(a)

print("b")
mino.print_matrix(b)

print("c")
mino.print_matrix(c)

print("d")
mino.print_matrix(d)

print("g")
mino.print_matrix(g)

print("a grad")
mino.print_matrix_grad(a)

print("b grad")
mino.print_matrix_grad(b)

print("d grad")
mino.print_matrix_grad(d)
 