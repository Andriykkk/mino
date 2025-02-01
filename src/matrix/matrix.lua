package.path = package.path .. ";./error_handling/?.lua"
local error_handling = require('error_handling')
local matrix = {}

function broadcast_values(self, other)
    if self.dims[1] == other.dims[1] and self.dims[2] == other.dims[2] then
        return {1, self.dims[1], 1, self.dims[2]}
    end
    shape = {}
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

    return shape
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
            return nil
        end
    end,
    
    -- TODO: remake indices so it work with multidimensional indices
    __newindex = function(self, key, value)
        if type(key) == "number" and key <= self.dims[1] * self.dims[2] and type(value) == "number" then
            self.data[key] = value
        else
            error_handling.show_error("Invalid key or value for matrix.")
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

        local result = matrix.new({dims = {self.dims[1], self.dims[2]}})
        for i = 1, shape[1] do
            for ii = 1, shape[2] do
                for j = 1, shape[3] do
                    for jj = 1, shape[4] do
                        local row = (i - 1) * shape[2] + ii
                        local col = (j - 1) * shape[4] + jj

                        result[(row - 1) * self.dims[2] + col] = self[(row - 1) * self.dims[2] + col] + other.data[(ii - 1) * other.dims[2] + jj]
                    end
                end
            end
        end
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

        if self.dims[1] ~= other.dims[1] or self.dims[2] ~= other.dims[2] then
            error_handling.show_error("Matrices are not compatible for subtraction.")
        end

        local result = matrix.new({dims = {self.dims[1], self.dims[2]}})
        for i = 1, self.dims[1] * self.dims[2] do
            result.data[i] = self.data[i] - other.data[i]
        end
        return result
    end,

    __mul = function(self, other)
        if type(other) == "number" then
            local result = matrix.new({dims = {self.dims[1], self.dims[2]}})
            for i = 1, self.dims[1] * self.dims[2] do
                result.data[i] = self.data[i] * other
            end
            return result
        end
        
        if type(self) == "number" then
            local result = matrix.new({dims = {other.dims[1], other.dims[2]}})
            for i = 1, other.dims[1] * other.dims[2] do
                result.data[i] = other.data[i] * self
            end
            return result
        end

        if self.dims[1] ~= other.dims[1] or self.dims[2] ~= other.dims[2] then
            error_handling.show_error("Matrices are not compatible for multiplication.")
        end

        local result = matrix.new({dims = {self.dims[1], self.dims[2]}})
        for i = 1, self.dims[1] * self.dims[2] do
            result.data[i] = self.data[i] * other.data[i]
        end
        return result
    end

}

function matrix.new(params)
    local matrix = {}
    matrix.dims = {0, 0}
    matrix.data = {}

    setmetatable(matrix, matrix_mt)

    local fill_value = 0

    if params.dims == nil and params.data == nil then
        error_handling.show_error("Invalid params for matrix.")
    end

    if params.dims ~= nil then
        matrix.dims[1] = params.dims[1]
        matrix.dims[2] = params.dims[2]
    end 

    if type(params.data) == "table" then
        if params.dims ~= nil 
        and (params.dims[1] == #params.data and params.dims[2] == #params.data[1])
        or (params.dims[1] * params.dims[2] == #params.data)
        then
            copy_table_to_matrix(matrix, params.data)
            return matrix
        else
            error_handling.show_error("Invalid data for matrix, dimensions do not match.")
        end         
    end

    if type(params.data) == "number" then
        fill_value = params.data
    end

    if type(params.data) == "function" then
        -- TODO add generation of matrix numbers from function
    end

    for i = 1, matrix.dims[1] * matrix.dims[2] do
        matrix.data[i] = fill_value
    end

    return matrix
end 

function matmul_naive(self, other)
    if self.dims[2] ~= other.dims[1] then
        error_handling.show_error("Matrices are not compatible for multiplication.")
    end

    local result = matrix.new({dims = {self.dims[1], other.dims[2]}})

    for i = 1, self.dims[1] do
        for j = 1, other.dims[2] do
            local sum = 0
            for k = 1, self.dims[2] do
                sum = sum + self.data[(i - 1) * self.dims[2] + k] * other.data[(k - 1) * other.dims[2] + j]
            end
            result.data[(i - 1) * other.dims[2] + j] = sum
        end
    end

    return result
end

matrix.matmul_naive = matmul_naive
matrix.matmul = matmul_naive

return matrix