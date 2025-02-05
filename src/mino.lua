package.path = package.path .. ";./layers/?.lua;./matrix/?.lua;./activations/?.lua;"
local layers = require('layers')
local matrix = require('matrix')
local activations = require('activations')
local utils = require('utils')
local loss = require('loss')

local mino = {}
mino.layers = layers
mino.matrix = matrix
mino.activations = activations
mino.utils = utils
mino.loss = loss


function copy_table_to_matrix(matrix, data)
    for i = 1, #data do
        if type(data[i]) == "table" then
            if matrix.dims[2] <= #data[i] then
                print("Invalid data for matrix, dimensions do not match")
                os.exit(1)
            end
            for j = 1, #data[i] do
                if type(data[i][j]) ~= "number" then
                    print("Invalid data for matrix, data is not a number")
                    os.exit(1)
                end
                matrix.data[(i - 1) * matrix.dims[2] + j] = data[i][j]
            end
        else
            if type(data[i]) ~= "number" then
                print("Invalid data for matrix, data is not a number")
                os.exit(1)
            end
            matrix.data[i] = data[i]
        end
    end
end

function mino.print_matrix(matrix)
    if matrix == nil or matrix.data == nil then
        print("Matrix is empty")
        return
    end
    for i = 1, matrix.dims[1] do
        io.write("{ ")
        for j = 1, matrix.dims[2] do
            io.write(matrix.data[(i - 1) * matrix.dims[2] + j])
            if j ~= matrix.dims[2] then
                io.write(", ")
            end
        end
        io.write(" }")
        if i ~= matrix.dims[1] then
            io.write("\n")
        end
    end
    io.write(", shape: { " .. matrix.dims[1] .. ", " .. matrix.dims[2] .. " }")
    
    io.write("\n")
end

function mino.print_matrix_grad(matrix)
    if matrix.required_grad ~= true then
        print("Matrix does not have a gradient")
        return
    end

    for i = 1, matrix.dims[1] do
        io.write("{ ")
        for j = 1, matrix.dims[2] do
            io.write(matrix.grad[(i - 1) * matrix.dims[2] + j])
            if j ~= matrix.dims[2] then
                io.write(", ")
            end
        end
        io.write(" }")
        if i ~= matrix.dims[1] then
            io.write("\n")
        end
    end
    io.write(", shape: { " .. matrix.dims[1] .. ", " .. matrix.dims[2] .. " }")
    
    io.write("\n")
end


-- call libraries
-- layers
function mino.layers.Linear(params)
    return layers.linear.new(params, mino)
end

-- matrix
function mino.Matrix(params)
    return matrix.new(params, mino)
end
function mino.matmul(self, other)
    return matrix.matmul(self, other)
end



return mino