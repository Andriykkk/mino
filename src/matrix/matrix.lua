package.path = package.path .. ";./error_handling/?.lua"
local error_handling = require('error_handling')
local addition_kernel = require('matrix/kernels/addition')
local subtraction_kernel = require('matrix/kernels/subtraction')
local multiplication_kernel = require('matrix/kernels/multiplication')
local matmul_kernel = require('matrix/kernels/matmul')
local matrix = {}

BIG = 1
SMALL = 0

function broadcast_values(self, other)
    if self.dims[1] == other.dims[1] and self.dims[2] == other.dims[2] then
        return {1, self.dims[1], 1, self.dims[2]}
    end
    shape = {}
    shape[5] = 1

    if self.dims[1] <= other.dims[1] and self.dims[2] <= other.dims[2] then
        local temp = self
        self = other
        other = temp
        shape[5] = 0
    end

    if other.dims[1] ~= 1 and other.dims[2] ~= 1 then
        return false
    end

    if self.dims[1] >= other.dims[1] and self.dims[1] % other.dims[1] == 0 then
        shape[1] = self.dims[1] / other.dims[1]
        shape[2] = other.dims[1]
    else
        return false
    end

    if self.dims[2] >= other.dims[2] and self.dims[2] % other.dims[2] == 0 then
        shape[3] = self.dims[2] / other.dims[2]
        shape[4] = other.dims[2]
    else
        return false
    end

    if shape[5] == SMALL then
        local temp = self
        self = other
        other = temp
    end

    return shape
end

function add_backward(self, respect)
    local operand1 = self.operand1
    local operand2 = self.operand2

    if operand1.required_grad == true then
        shape = broadcast_values(operand1, respect)
        if shape[5] == BIG then
            addition_kernel.run_big_back(operand1.grad, respect.data, operand1.dims[2], respect.dims[2], shape[1], shape[2], shape[3], shape[4])
        else
            addition_kernel.run_small_back(respect.data, operand1.grad, respect.dims[2], operand1.dims[2], shape[1], shape[2], shape[3], shape[4])
        end
    end

    if operand1.backward then
        operand1:backward(respect)
    end

    if operand2.required_grad == true then
        shape = broadcast_values(operand2, respect)
        if shape[5] == BIG then
            addition_kernel.run_big_back(operand2.grad, respect.data, operand2.dims[2], respect.dims[2], shape[1], shape[2], shape[3], shape[4])
        else
            addition_kernel.run_small_back(respect.data, operand2.grad, respect.dims[2], operand2.dims[2], shape[1], shape[2], shape[3], shape[4])
        end
    end

    if operand2.backward then
        operand2:backward(respect)
    end
end

function subtract_backward(self, respect)
    local operand1 = self.operand1
    local operand2 = self.operand2

    if operand1.required_grad == true then
        shape = broadcast_values(operand1, respect)
        if shape[5] == BIG then
            addition_kernel.run_big_back(operand1.grad, respect.data, operand1.dims[2], respect.dims[2], shape[1], shape[2], shape[3], shape[4])
        else
            addition_kernel.run_small_back(respect.data, operand1.grad, respect.dims[2], operand1.dims[2], shape[1], shape[2], shape[3], shape[4])
        end
    end

    if operand1.backward then
        operand1:backward(respect)
    end

    for i = 1, self.dims[1] * self.dims[2] do
        respect[i] = -respect[i]
    end

    if operand2.required_grad == true then
        shape = broadcast_values(operand2, respect)
        if shape[5] == BIG then
            addition_kernel.run_big_back(operand2.grad, respect.data, operand2.dims[2], respect.dims[2], shape[1], shape[2], shape[3], shape[4])
        else
            addition_kernel.run_small_back(respect.data, operand2.grad, respect.dims[2], operand2.dims[2], shape[1], shape[2], shape[3], shape[4])
        end
    end

    if operand2.backward then
        operand2:backward(respect)
    end
end

function mul_backward(self, respect)
    local operand1 = self.operand1
    local operand2 = self.operand2

    if type(operand2) == "number" then
        for i = 1, self.dims[1] * self.dims[2] do
            respect[i] = respect[i] * operand2
        end

        if operand1.backward then
            operand1:backward(respect)
        end
        return
    end

    if operand1.required_grad == true then
        shape = broadcast_values(operand2, respect)
        local result = {}
        for i = 1, operand2.dims[1] * operand2.dims[2] do
            result[i] = 0
        end
        if shape[5] == BIG then
            multiplication_kernel.run_big(operand2.data, respect.data, result, operand2.dims[2], respect.dims[2], shape[1], shape[2], shape[3], shape[4])
        else
            multiplication_kernel.run_small(respect.data, operand2.data, result, respect.dims[2], operand2.dims[2], shape[1], shape[2], shape[3], shape[4])
        end

        shape = broadcast_values(operand1, operand2)
        if shape[5] == BIG then
            addition_kernel.run_big_back(operand1.grad, result, operand1.dims[2], operand2.dims[2], shape[1], shape[2], shape[3], shape[4])
        else
            addition_kernel.run_small_back(result, operand1.grad, operand2.dims[2], operand1.dims[2], shape[1], shape[2], shape[3], shape[4])
        end
    end

    if operand1.backward then
        operand1:backward(respect)
    end

    if operand2.required_grad == true then
        shape = broadcast_values(operand1, respect)
        local result = {}
        for i = 1, operand1.dims[1] * operand1.dims[2] do
            result[i] = 0
        end
        if shape[5] == BIG then
            multiplication_kernel.run_big(operand1.data, respect.data, result, operand1.dims[2], respect.dims[2], shape[1], shape[2], shape[3], shape[4])
        else
            multiplication_kernel.run_small(respect.data, operand1.data, result, respect.dims[2], operand1.dims[2], shape[1], shape[2], shape[3], shape[4])
        end

        shape = broadcast_values(operand2, operand1)
        if shape[5] == BIG then
            addition_kernel.run_big_back(operand2.grad, result, operand2.dims[2], operand1.dims[2], shape[1], shape[2], shape[3], shape[4])
        else
            addition_kernel.run_small_back(result, operand2.grad, operand1.dims[2], operand2.dims[2], shape[1], shape[2], shape[3], shape[4])
        end
    end

    if operand2.backward then
        operand2:backward(respect)
    end

end

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
    
    __add = function(self, other)
        if type(other) == "number" then
            local result = matrix.new({dims = {self.dims[1], self.dims[2]}})
            for i = 1, self.dims[1] * self.dims[2] do
                result.data[i] = self.data[i] + other
            end
            return result
        end
        
        if type(self) == "number" then
            local result = matrix.new({dims = {other.dims[1], other.dims[2]}})
            for i = 1, other.dims[1] * other.dims[2] do
                result.data[i] = other.data[i] + self
            end
            return result
        end
        
        shape = broadcast_values(self, other)
        if shape == false then
            error_handling.show_error("Matrices dimensions are not compatible for addition.")
        end
        
        if shape[5] == SMALL then
            local temp = self
            self = other
            other = temp
        end

        local result = matrix.new({dims = {self.dims[1], self.dims[2]}})
        result.backward = add_backward
        result.operand1 = self
        result.operand2 = other

        addition_kernel.run_big(self.data, other.data, result.data, self.dims[2], other.dims[2], shape[1], shape[2], shape[3], shape[4])
        return result
    end,

    __sub = function(self, other)
        if type(other) == "number" then
            local result = matrix.new({dims = {self.dims[1], self.dims[2]}})
            for i = 1, self.dims[1] * self.dims[2] do
                result.data[i] = self.data[i] - other
            end
            return result
        end
        
        if type(self) == "number" then
            local result = matrix.new({dims = {other.dims[1], other.dims[2]}})
            for i = 1, other.dims[1] * other.dims[2] do
                result.data[i] = other.data[i] - self
            end
            return result
        end

        shape = broadcast_values(self, other)
        if shape == false then
            error_handling.show_error("Matrices dimensions are not compatible for subtraction.")
        end
        
        if shape[5] == SMALL then
            local temp = self
            self = other
            other = temp
        end
        
        local result = matrix.new({dims = {self.dims[1], self.dims[2]}})
        result.backward = subtract_backward
        result.operand1 = self
        result.operand2 = other

        if shape[5] == SMALL then
            result.operand1 = other
            result.operand2 = self

        end
        subtraction_kernel.run_big(self.data, other.data, result.data, self.dims[2], other.dims[2], shape[1], shape[2], shape[3], shape[4])
        return result
    end,

    __mul = function(self, other)
        -- add backward for simple multiplication
        if type(other) == "number" then
            local result = matrix.new({dims = {self.dims[1], self.dims[2]}})
            for i = 1, self.dims[1] * self.dims[2] do
                result.data[i] = self.data[i] * other
            end
            result.backward = mul_backward
            result.operand1 = self
            result.operand2 = other
            return result
        end
        
        if type(self) == "number" then
            local result = matrix.new({dims = {other.dims[1], other.dims[2]}})
            for i = 1, other.dims[1] * other.dims[2] do
                result.data[i] = other.data[i] * self
            end
            result.backward = mul_backward
            result.operand1 = other
            result.operand2 = self
            return result
        end

        shape = broadcast_values(self, other)
        if shape == false then
            error_handling.show_error("Matrices dimensions are not compatible for addition.")
        end
        
        if shape[5] == SMALL then
            local temp = self
            self = other
            other = temp
        end

        local result = matrix.new({dims = {self.dims[1], self.dims[2]}})
        result.backward = mul_backward
        result.operand1 = self
        result.operand2 = other
        multiplication_kernel.run_big(self.data, other.data, result.data, self.dims[2], other.dims[2], shape[1], shape[2], shape[3], shape[4])
        return result
    end 
}


function matrix.new(params)
    local mat = {}
    mat.dims = {0, 0}
    mat.data = {}
    mat.grad = {}
    mat.required_grad = true
    mat.copy = matrix.copy

    setmetatable(mat, matrix_mt)

    local fill_value = 0

    -- dimensions and data
    if params.dims == nil and params.data == nil then
        error_handling.show_error("Invalid params for matrix.")
    end

    if params.dims ~= nil then
        mat.dims[1] = params.dims[1]
        mat.dims[2] = params.dims[2]
    end 

    if type(params.data) == "table" then
        if params.dims ~= nil 
        and (params.dims[1] == #params.data and params.dims[2] == #params.data[1])
        or (params.dims[1] * params.dims[2] == #params.data)
        then
            copy_table_to_matrix(mat, params.data)
        else
            error_handling.show_error("Invalid data for matrix, dimensions do not match.")
        end         
    else
        -- fill matrix
        if type(params.data) == "number" then
            fill_value = params.data
        end

        if type(params.data) == "function" then
            -- TODO add generation of matrix numbers from function
        end

        for i = 1, mat.dims[1] * mat.dims[2] do
            mat.data[i] = fill_value
        end
    end

    -- required_grad
    if params.required_grad == false then
        mat.required_grad = false
    end

    if mat.required_grad == true then
        for i = 1, mat.dims[1] * mat.dims[2] do
            mat.grad[i] = 0
        end
    end

    return mat
end 

function matrix:copy(params)
    if params == nil then
        params = {}
    end
    local copy = matrix.new({dims = {self.dims[1], self.dims[2]}, data = self.data, required_grad = self.required_grad})

    for i = 1, self.dims[1] * self.dims[2] do
        copy.data[i] = self.data[i]
    end

    if params.copy_grad == true then
        for i = 1, self.dims[1] * self.dims[2] do
            copy.grad[i] = self.grad[i]
        end
    end

    return copy
end

function matrix.matmul(self, other)
    if self.dims[2] ~= other.dims[1] then
        error_handling.show_error("Matrices are not compatible for multiplication.")
    end

    local result = matrix.new({dims = {self.dims[1], other.dims[2]}})

    result.backward = matmul_backward
    result.operand1 = self
    result.operand2 = other

    matmul_kernel.matmul_naive(self, other, result)

    return result
end

return matrix