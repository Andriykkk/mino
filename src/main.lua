package.path = package.path .. ";./error_handling/?.lua;./activations/?.lua;./utils/?.lua;./loss/?.lua;"
local error_handling = require('error_handling')
local mino = require('mino')
local activations = require('activations')

 
local a = mino.Matrix({dims = {2, 2}, data = 2})
local b = mino.Matrix({dims = {1, 2}, data = 1})
local d = mino.Matrix({dims = {2, 2}, data = 1})

g = d + b
c = a + g

target = mino.Matrix({data = {{1}, {1}}})

loss = mino.loss.cross_entropy(c, target)
loss:backward()

mino.print_matrix(c)
mino.print_matrix(target)

print("a")
mino.print_matrix(a)
print("b")
mino.print_matrix(b)
print("d")
mino.print_matrix(d)
print("c")
mino.print_matrix(c)
print("loss")
mino.print_matrix(loss)

print("a")
mino.print_matrix_grad(a)
print("b")
mino.print_matrix_grad(b)
print("d")
mino.print_matrix_grad(d)
print("loss")
mino.print_matrix_grad(loss)