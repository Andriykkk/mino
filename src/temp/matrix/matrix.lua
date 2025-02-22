package.path = package.path .. ";./error_handling/?.lua"
local error_handling = require('error_handling')
local addition_kernel = require('matrix/kernels/addition')
local subtraction_kernel = require('matrix/kernels/subtraction')
local multiplication_kernel = require('matrix/kernels/multiplication')
local matmul_kernel = require('matrix/kernels/matmul')
local matrix = {}

function matmul_backward(self, respect)
    local operand1 = self.operand1
    local operand2 = self.operand2

    local result
    if operand1.required_grad == true then
        local operand_t = matrix.new({dims = {operand2.dims[2], operand2.dims[1]}})
        for i = 1, self.dims[1] do
            for j = 1, self.dims[2] do
                operand_t.data[(j - 1) * operand2.dims[1] + i] = operand2.data[(i - 1) * operand2.dims[2] + j]
            end
        end
        result = matrix.new({dims = {operand_t.dims[1], respect.dims[2]}})
        matmul_kernel.matmul_naive(respect, operand_t, result)

        for i = 1, result.dims[1] * result.dims[2] do
            operand1.grad[i] = operand1.grad[i] + result.data[i]
        end
    end

    if operand1.backward then
        operand1:backward(result)
    end

    if operand2.required_grad == true then
        local operand_t = matrix.new({dims = {operand1.dims[2], operand1.dims[1]}})
        for i = 1, self.dims[1] do
            for j = 1, self.dims[2] do
                operand_t.data[(j - 1) * operand1.dims[1] + i] = operand1.data[(i - 1) * operand1.dims[2] + j]
            end
        end
        result = matrix.new({dims = {operand_t.dims[1], respect.dims[2]}})
        matmul_kernel.matmul_naive(operand_t, respect, result)

        for i = 1, result.dims[1] * result.dims[2] do
            operand2.grad[i] = operand2.grad[i] + result.data[i]
        end
    end

    if operand2.backward then
        operand2:backward(result)
    end
end

local matrix_mt = {
    -- TODO: remake indexing when i create multidimensional matrices it should return smaller and smaller matrices
    -- first return one matrix with stride, then another and so as much indices, that just spend a little time for matrix creating with stride
    __index = function(self, key)
        if type(key) == "number" then
            -- return setmetatable({data = self.data[key]}, matrix_mt)
            return self.data[key]
        elseif key == "T" then
            local result = matrix.new({dims = {self.dims[2], self.dims[1]}})
            for i = 1, self.dims[1] do
                for j = 1, self.dims[2] do
                    result.data[(j - 1) * self.dims[1] + i] = self.data[(i - 1) * self.dims[2] + j]
                end
            end
            return result
        elseif key == "shape" then
            return self.dims
        else 
            return rawget(self, key)
        end
    end,
    
    -- TODO: remake indices so it work with multidimensional indices
    __newindex = function(self, key, value)
        if type(key) == "number" and key <= self.dims[1] * self.dims[2] and type(value) == "number" then
            self.data[key] = value
        else
            rawset(self, key, value)
        end
    end,

}

-- operations
function log_backward(self, respect)
    local operand1 = self.operand1
    local result = matrix.new({dims = {operand1.dims[1], operand1.dims[2]}})
    for i = 1, operand1.dims[1] * operand1.dims[2] do
        result.data[i] = respect.data[i] / operand1.data[i]
    end
    if operand1.required_grad == true then
        for i = 1, operand1.dims[1] * operand1.dims[2] do
            operand1.grad[i] = operand1.grad[i] + result.data[i]
        end
    end
    if operand1.backward then
        operand1:backward(result)
    end
end
function matrix.log(self)
    local result = matrix.new({dims = {self.dims[1], self.dims[2]}})
    for i = 1, self.dims[1] * self.dims[2] do
        result.data[i] = math.log(self.data[i])
    end
    result.operand1 = self
    result.backward = log_backward
    return result
end

function sum_backward(self)
    local operand1 = self.operand1
    local respect = matrix.new({dims = {operand1.dims[1], operand1.dims[2]}, data=1})

    if operand1.required_grad == true then
        for i = 1, operand1.dims[1] * operand1.dims[2] do
            operand1.grad[i] = operand1.grad[i] + respect.data[1]
        end
    end

    if operand1.backward then
        operand1:backward(respect)
    end
end
function matrix.sum(self)
    local result = matrix.new({dims = {1, 1}})
    for i = 1, self.dims[1] * self.dims[2] do
        result.data[1] = result.data[1] + self.data[i]
    end
    result.operand1 = self
    result.backward = sum_backward
    return result
end

return matrix