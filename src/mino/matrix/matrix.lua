local error_handling = require('error_handling')
local add_tables = require('add_tables')
local transpose_tables = require('transpose_tables')
local mul_tables = require('mul_tables')
local matmul_tables = require('matmul_tables')
local utils = require('utils')
local matrix = {}

-- UTILS
local function max(a, b)
    if a > b then
        return a
    else
        return b
    end
end
local function min(a, b)
    if a < b then
        return a
    else
        return b
    end
end
local function print_table(table)
    for i = 1, #table do
        io.write(string.format("%.4f", table[i]))
        if i ~= #table then
            io.write(", ")
        end
    end
    io.write("\n")
end
local function create_result(self, other)
    -- when one of matrices is scalar
    if type(other) == "number" then
        local dimensions = {}
        for i = 1, #self.sub_dims do
            dimensions[i] = self.sub_dims[i]
        end
        table.insert(dimensions, self.dims[1])
        table.insert(dimensions, self.dims[2])
        return matrix.new({dims = dimensions})
    end

    -- check dimensions
    if not (self.dims[1] == other.dims[1] or self.dims[1] == 1 or other.dims[1] == 1) then
        error_handling.dimension_error(self, other)
    elseif not (self.dims[2] == other.dims[2] or self.dims[2] == 1 or other.dims[2] == 1) then
        error_handling.dimension_error(self, other)
    end

    -- make sub_dim of the same size
    local dimensions = {}
    if #self.sub_dims ~= #other.sub_dims then
        if #self.sub_dims > #other.sub_dims then
            for i = 1, #self.sub_dims - #other.sub_dims do
                table.insert(dimensions, 1)
            end
            for i = 1, #other.sub_dims do
                table.insert(dimensions, other.sub_dims[i])
            end
            table.insert(dimensions, other.dims[1])
            table.insert(dimensions, other.dims[2])
            other:view(dimensions)
        else
            for i = 1, #other.sub_dims - #self.sub_dims do
                table.insert(dimensions, 1)
            end
            for i = 1, #self.sub_dims do
                table.insert(dimensions, other.sub_dims[i])
            end
            table.insert(dimensions, self.dims[1])
            table.insert(dimensions, self.dims[2])
            self:view(dimensions)
        end
    end

    -- copy biggest dims
    local dimensions = {}
    for i = 1, #self.sub_dims do
        if not(self.dims[1] ~= other.dims[1] or (self.dims[1] ~= 1 or other.dims[1] ~= 1)) then
            error_handling.dimension_error(self, other)
        end
        dimensions[i] = max(self.sub_dims[i], other.sub_dims[i])
    end
    table.insert(dimensions, max(self.dims[1], other.dims[1]))
    table.insert(dimensions, max(self.dims[2], other.dims[2]))

    local result = matrix.new({dims = dimensions})
    return result
end
local function create_result_matmul(self, other)
    -- check dimensions
    if self.dims[2] ~= other.dims[1] then
        error_handling.dimension_error(self, other)
    end

    -- make sub_dim of the same size
    local dimensions = {}
    if #self.sub_dims ~= #other.sub_dims then
        if #self.sub_dims > #other.sub_dims then
            for i = 1, #self.sub_dims - #other.sub_dims do
                table.insert(dimensions, 1)
            end
            for i = 1, #other.sub_dims do
                table.insert(dimensions, other.sub_dims[i])
            end
            table.insert(dimensions, other.dims[1])
            table.insert(dimensions, other.dims[2])
            other:view(dimensions)
        else
            for i = 1, #other.sub_dims - #self.sub_dims do
                table.insert(dimensions, 1)
            end
            for i = 1, #self.sub_dims do
                table.insert(dimensions, other.sub_dims[i])
            end
            table.insert(dimensions, self.dims[1])
            table.insert(dimensions, self.dims[2])
            self:view(dimensions)
        end
    end

    -- copy biggest dims
    local dimensions = {}
    for i = 1, #self.sub_dims do
        dimensions[i] = max(self.sub_dims[i], other.sub_dims[i])
    end
    table.insert(dimensions, self.dims[1])
    table.insert(dimensions, other.dims[2])

    local result = matrix.new({dims = dimensions})
    return result
end
local function sub_dims_divider(dim_index, self_s, other_s, result_s, func, self, other, result, params)
    if dim_index <= #self.sub_dims and self.sub_dims[dim_index] ~= nil then
        for i = 0, result.sub_dims[dim_index] - 1 do
            local self_stride = 0
            local other_stride = 0
            if self.sub_dims[dim_index] ~= 1 then
                self_stride = self_s + self.strides[dim_index][2] * i
            else
                self_stride = self_s
            end
            if other.sub_dims[dim_index] ~= 1 then
                other_stride = other_s + other.strides[dim_index][2] * i
            else
                other_stride = other_s
            end
            sub_dims_divider(dim_index + 1, self_stride, other_stride, result_s + result.strides[dim_index][2] * i, func, self, other, result, params)
        end
    else
        func(self, other, result, self_s, other_s, result_s, params)
    end
end
local function sub_dims_divider_one(dim_index, self_s, other_s, func, self, other)
    if dim_index <= #other.sub_dims then
        for i = 0, other.sub_dims[dim_index] - 1 do
            local self_stride = 0
            local other_stride = 0
            if self.sub_dims[dim_index] ~= 1 then
                self_stride = self_s + self.strides[dim_index][2] * i
            else
                self_stride = self_s
            end
            if other.sub_dims[dim_index] ~= 1 then
                other_stride = other_s + other.strides[dim_index][2] * i
            else
                other_stride = other_s
            end
            sub_dims_divider_one(dim_index + 1, self_stride, other_stride, func, self, other)
        end
    else
        func(self, other, self_s, other_s)
    end
end
-- UTILS

-- BACKWARD
local function add_backward(self, respect)
    local operand1 = self.operand1
    local operand2 = self.operand2

    if operand1.required_grad == true then
        operand1.values = operand1.grad
        respect.values = respect.data
        sub_dims_divider_one(1, 0, 0, add_tables.one, operand1, respect)
    end

    if operand1.backward then
        operand1:backward(respect)
    end


    if operand2 ~= nil then
        if operand2.required_grad == true then
            operand2.values = operand2.grad
            respect.values = respect.data
            sub_dims_divider_one(1, 0, 0, add_tables.one, operand2, respect)
        end

        if operand2.backward then
            operand2:backward(respect)
        end
    end
end
local function sub_backward(self, respect)
    local operand1 = self.operand1
    local operand2 = self.operand2

    if operand1.required_grad == true then
        operand1.values = operand1.grad
        respect.values = respect.data
        sub_dims_divider_one(1, 0, 0, add_tables.one, operand1, respect)
    end

    if operand1.backward then
        operand1:backward(respect)
    end

    if operand2 ~= nil then
        for i = 1, #respect.data do
            respect.data[i] = -respect.data[i]
        end

        if operand2.required_grad == true then
            operand2.values = operand2.grad
            respect.values = respect.data
            sub_dims_divider_one(1, 0, 0, add_tables.one, operand2, respect)
        end

        if operand2.backward then
            operand2:backward(respect)
        end
    end
end
local function unm_backward(self, respect)
    local operand1 = self.operand1

    for i = 1, #respect.data do
        respect.data[i] = -respect.data[i]
    end

    if operand1.required_grad == true then
        operand1.values = operand1.grad
        respect.values = respect.data
        sub_dims_divider_one(1, 0, 0, add_tables.one, operand1, respect)
    end

    if operand1.backward then
        operand1:backward(respect)
    end
end
local function mul_backward(self, respect)
    local operand1 = self.operand1
    local operand2 = self.operand2

    if operand2 == nil then
        for i = 1, #respect.data do
            respect.data[i] = respect.data[i] * operand1.data[i]
        end

        if operand1.backward then
            operand1:backward(respect)
        end
        return
    end
    
    local result = operand2:copy({data = 0})
    operand2.values = operand2.grad
    respect.values = respect.data
    result.values = result.data
    sub_dims_divider(1, 0, 0, 0, mul_tables.back, operand2, respect, result)

    if operand1.required_grad == true then
        operand2.values = operand2.grad
        sub_dims_divider_one(1, 0, 0, add_tables.one, operand2, respect)
    end

    if operand1.backward then
        operand1:backward(result)
    end

    result = operand1:copy({data = 0})
    operand1.values = operand1.grad
    result.values = result.data
    sub_dims_divider(1, 0, 0, 0, mul_tables.back, operand1, respect, result)

    if operand2.required_grad == true then
        operand1.values = operand1.grad
        sub_dims_divider_one(1, 0, 0, add_tables.one, operand1, respect)
    end

    if operand2.backward then
        operand2:backward(result)
    end
end
local function matmul_backward(self, respect)
    local operand1 = self.operand1
    local operand2 = self.operand2
    local shape = operand2:shape()

    -- transpose and matmul operand 2 and respect to get result
    local operand_t = operand2:copy():T(#shape, #shape - 1)
    
    local result_dims = operand_t:shape()
    result_dims[#result_dims  - 1] = respect.dims[1]
    local result = matrix.new({dims = result_dims})

    operand_t.values = operand_t.data
    result.values = result.data
    respect.values = respect.data

    sub_dims_divider(1, 0, 0, 0, matmul_tables.result, respect, operand_t, result)

    if operand1.required_grad == true then
        operand1.values = operand1.grad
        result.values = result.data
        sub_dims_divider_one(1, 0, 0, add_tables.one, operand1, result)
    end

    if operand1.backward then
        operand1:backward(result)
    end

    local shape = operand1:shape()

    -- transpose and matmul operand 1 and respect to get result
    local shape = operand1:shape()
    local operand_t = operand1:copy():T(#shape, #shape - 1)

    local result_dims = operand_t:shape()
    result_dims[#result_dims] = respect.dims[2]
    local result = matrix.new({dims = result_dims})

    operand_t.values = operand_t.data
    result.values = result.data
    respect.values = respect.data

    sub_dims_divider(1, 0, 0, 0, matmul_tables.result, operand_t, respect, result)

    if operand2.required_grad == true then
        operand2.values = operand2.grad
        result.values = result.data
        sub_dims_divider_one(1, 0, 0, add_tables.one, operand2, result)
    end

    if operand2.backward then
        operand2:backward(result)
    end
end
local function sum_backward(self)
    local operand1 = self.operand1
    local result = self.operand1:copy({data = 1})

    if operand1.required_grad == true then
        for i = 1, #result.data do
            operand1.grad[i] = operand1.grad[i] + result.data[i]
        end
    end

    if operand1.backward then
        operand1:backward(result)
    end
end
local function log_backward(self, respect)
    local operand1 = self.operand1
    local result = operand1:copy({data = 0})

    for i = 1, #result.size do
        result.data[i] = respect.data[i] / operand1.data[i]
    end

    if operand1.required_grad == true then
        for i = 1, #result.data do
            operand1.grad[i] = operand1.grad[i] + result.data[i]
        end
    end

    if operand1.backward then
        operand1:backward(result)
    end
end
-- BACKWARD

local matrix_mt = {
    -- TODO: remake indexing when i create multidimensional matrices it should return smaller and smaller matrices
    -- first return one matrix with stride, then another and so as much indices, that just spend a little time for matrix creating with stride
    -- __index = function(self, key)
    --     if type(key) == "number" then
    --         -- return setmetatable({data = self.data[key]}, matrix_mt)
    --         return self.data[key]
    --     elseif key == "T" then
    --         local result = matrix.new({dims = {self.dims[2], self.dims[1]}})
    --         for i = 1, self.dims[1] do
    --             for j = 1, self.dims[2] do
    --                 result.data[(j - 1) * self.dims[1] + i] = self.data[(i - 1) * self.dims[2] + j]
    --             end
    --         end
    --         return result
    --     elseif key == "shape" then
    --         return self.dims
    --     else 
    --         return rawget(self, key)
    --     end
    -- end,
    
    -- TODO: remake indices so it work with multidimensional indices
    -- __newindex = function(self, key, value)
    --     if type(key) == "number" and key <= self.dims[1] * self.dims[2] and type(value) == "number" then
    --         self.data[key] = value
    --     else
    --         rawset(self, key, value)
    --     end
    -- end,
    __index = matrix,
    __add = function(self, other)
        -- UTILS
        local function add_tables(self, other, result, self_pos, other_pos, res_pos)
            for i = 0, result.dims[1] - 1 do
                for j = 0, result.dims[2] - 1 do
                    local self_i = i % self.dims[1]
                    local self_j = j % self.dims[2]

                    local other_i = i % other.dims[1]
                    local other_j = j % other.dims[2]

                    local self_val = self.data[self_pos + self_i * self.dims[2] + self_j + 1]
                    local other_val = other.data[other_pos + other_i * other.dims[2] + other_j + 1]

                    result.data[res_pos + i * result.dims[2] + j + 1] = self_val + other_val
                end
            end
        end
        -- UTILS
        if type(self) == "number" then
            local temp = self
            self = other
            other = temp
        end

        local result = create_result(self, other)
        
        if type(other) == "number" then
            for i = 1, result.size do
                result.data[i] = self.data[i] + other
            end

            result.backward = add_backward
            result.operand1 = self
            return result
        end

        sub_dims_divider(1, 0, 0, 0, add_tables, self, other, result)

        result.backward = add_backward
        result.operand1 = self
        result.operand2 = other

        return result
    end,
    __sub = function(self, other)
        -- UTILS
        local function subtract_tables(self, other, result, self_pos, other_pos, res_pos)
            for i = 0, result.dims[1] - 1 do
                for j = 0, result.dims[2] - 1 do
                    local self_i = i % self.dims[1]
                    local self_j = j % self.dims[2]

                    local other_i = i % other.dims[1]
                    local other_j = j % other.dims[2]

                    local self_val = self.data[self_pos + self_i * self.dims[2] + self_j + 1]
                    local other_val = other.data[other_pos + other_i * other.dims[2] + other_j + 1]

                    result.data[res_pos + i * result.dims[2] + j + 1] = self_val - other_val
                end
            end
        end
        -- UTILS
        if type(self) == "number" then
            error_handling.show_error("Can't subtract number from matrix.")
        end

        local result = create_result(self, other)
        
        if type(other) == "number" then
            for i = 1, result.size do
                result.data[i] = self.data[i] - other
            end

            result.backward = sub_backward
            result.operand1 = self
            return result
        end

        sub_dims_divider(1, 0, 0, 0, subtract_tables, self, other, result)

        result.backward = sub_backward
        result.operand1 = self
        result.operand2 = other

        return result
    end,
    __unm = function(self)
        local dims = {}
        for i = 1, #self.sub_dims do
            table.insert(dims, self.sub_dims[i])
        end
        table.insert(dims, self.dims[1])
        table.insert(dims, self.dims[2])
        local result = matrix.new({dims = dims})

        for i = 1, result.size do
            result.data[i] = -self.data[i]
        end

        result.backward = sub_backward
        result.operand1 = self

        return result
    end,
    __mul = function(self, other)
        if type(self) == "number" then
            local temp = self
            self = other
            other = temp
        end

        local result = create_result(self, other)
        
        if type(other) == "number" then
            for i = 1, result.size do
                result.data[i] = self.data[i] * other
            end

            result.backward = mul_backward
            result.operand1 = self
            return result
        end

        self.values = self.data
        other.values = other.data
        result.values = result.data
        sub_dims_divider(1, 0, 0, 0, mul_tables.result, self, other, result)

        result.backward = mul_backward
        result.operand1 = self
        result.operand2 = other

        return result
    end
}

local function matmul_naive(self, other)
    if type(other) == "number" then
        error_handling.show_error("Can't multiply matrix with number.")
    end

    local result = create_result_matmul(self, other)

    self.values = self.data
    other.values = other.data
    result.values = result.data
    sub_dims_divider(1, 0, 0, 0, matmul_tables.result, self, other, result)

    result.backward = matmul_backward
    result.operand1 = self
    result.operand2 = other

    return result
end

-- OPERATIONS
function matrix:matmul(other)
    return matmul_naive(self, other)
end
function matrix:T(dim1, dim2)
    local shape = self:shape()
    dim1 = dim1 or #shape
    dim2 = dim2 or 1

    assert(dim1 >= 1 and dim1 <= #shape, "dim1 is out of range")
    assert(dim2 >= 1 and dim2 <= #shape, "dim2 is out of range")
    assert(dim1 ~= dim2, "dim1 and dim2 must be different")

    local new_dims = {}
    for i = 1, #shape do
        new_dims[i] = shape[i]
    end
    new_dims[dim1] = shape[dim2]
    new_dims[dim2] = shape[dim1]

    local result = matrix.new({dims = new_dims, data = 0})

    result = transpose_tables.result(self, result, dim1, dim2, shape, new_dims)

    return result
end
function matrix:sum()
    local result = matrix.new({dims = {1, 1}})
    for i = 1, self.size do
        result.data[1] = result.data[1] + self.data[i]
    end
    result.operand1 = self
    result.backward = sum_backward
    return result
end
function matrix:log()
    local result = self:copy()

    for i = 1, result.size do
        result.data[i] = math.log(result.data[i])
    end

    result.operand1 = self
    result.backward = log_backward

    return result
end
-- function matrix:max()
--     local result = matrix.new({dims = {1, 1}})
--     result.data[1] = self.data[1]
--     for i = 2, self.size do
--         if self.data[i] > result.data[1] then
--             result.data[1] = self.data[i]
--         end
--     end
--     result.operand1 = self
--     result.backward = max_backward
--     return result
-- end
-- function matrix:argmax()
--     local result = matrix.new({dims = {1, 1}})
--     result.data[1] = 1
--     for i = 2, self.size do
--         if self.data[i] > self.data[result.data[1]] then
--             result.data[1] = i
--         end
--     end
--     result.operand1 = self
--     result.backward = argmax_backward
--     return result
-- end
-- OPERATIONS

function matrix.new(params)
    -- HELPERS
    local function flatten(tbl, data)
        for _, t in ipairs(tbl) do
            if type(t) == "table" then
                flatten(t, data)
            else
                table.insert(data, t)
            end
        end
    end
    local function get_dims(tbl, dims)
        if type(tbl) == "table" then
            table.insert(dims, #tbl)
            get_dims(tbl[1], dims)
        end
    end
    local function copy_dimensions(dimensions, mat)
        mat.dims[1] = dimensions[#dimensions - 1]
        mat.dims[2] = dimensions[#dimensions]
        for i = 1, #dimensions do
            if #dimensions - i >= 2 then
                table.insert(mat.sub_dims, dimensions[i])
            end
        end 
    end
    local function fill_table(table, size, value)
        for i = 1, size do
            table[i] = value
        end
    end
    function calculate_matrix_size(mat)
        local size = 1
        for i = 1, #mat.sub_dims do
            size = size * mat.sub_dims[i]
        end
        for i = 1, #mat.dims do
            size = size * mat.dims[i]
        end
        return size
    end
    -- HELPERS


    local mat = {}
    mat.dims = {}
    mat.sub_dims = {}
    mat.strides = {}
    mat.data = {}
    mat.grad = {}
    mat.required_grad = true
    mat.size = 0
    mat.td_size = 0
    setmetatable(mat, matrix_mt)

    if params.dims == nil and params.data == nil then
        error_handling.show_error("Invalid params for matrix.")
    end

    -- copy dims and copy data
    if params.dims ~= nil and params.data ~= nil then
        if #params.dims < 2 then
            error_handling.show_error("Invalid dims for matrix.")
        end

        -- copy dims
        mat.dims[1] = params.dims[#params.dims - 1]
        mat.dims[2] = params.dims[#params.dims]
        if #params.dims > 2 then
            for i = 1, #params.dims - 2 do
                table.insert(mat.sub_dims, params.dims[i])
            end
        end

        -- copy data
        if type(params.data) == "table" then
            flatten(params.data, mat.data)
            mat.size = calculate_matrix_size(mat)
            if #mat.data ~= mat.size then
                error_handling.show_error("Invalid data for matrix.")
            end
        elseif type(params.data) == "number" then
            mat.size = calculate_matrix_size(mat)
            fill_table(mat.data, mat.size, params.data)
        else
            error_handling.show_error("Invalid data for matrix.")
        end

        goto continue
    elseif params.dims == nil and params.data ~= nil and type(params.data) == "table" then
        local dimensions = {}
        get_dims(params.data, dimensions)
        copy_dimensions(dimensions, mat)

        mat.size = calculate_matrix_size(mat)
        flatten(params.data, mat.data)

        goto continue
    elseif params.dims ~= nil and params.data == nil then
        mat.dims[1] = params.dims[#params.dims - 1]
        mat.dims[2] = params.dims[#params.dims]
        if #params.dims > 2 then
            for i = 1, #params.dims - 2 do
                table.insert(mat.sub_dims, params.dims[i])
            end
        end
        mat.size = calculate_matrix_size(mat)
        fill_table(mat.data, mat.size, 0)
    else
        error_handling.show_error("Invalid params for matrix.")
    end

    ::continue::

    if params.fill_value ~= nil then
        if type(params.fill_value) == "number" then
            fill_table(mat.data, mat.size, params.fill_value)
        elseif params.fill_value == "ones" then
            fill_table(mat.data, mat.size, 1)
        elseif params.fill_value == "zeros" then
            fill_table(mat.data, mat.size, 0)
        elseif params.fill_value == "random" then
            for i = 1, mat.size do
                mat.data[i] = math.random()
            end
        elseif params.fill_value == "normal" then
            for i = 1, mat.size do
                mat.data[i] = math.random()
            end
        else
            error_handling.show_error("Invalid fill_value for matrix.")
        end
    end

    -- set strides and size
    mat.td_size = mat.dims[1] * mat.dims[2]
    if #mat.sub_dims > 0 then
        for i = 1, #mat.sub_dims do
            local big_stride = 1
            for j = 1, i do
                big_stride = big_stride * mat.sub_dims[j]
            end
            local small_stride = 1
            for j = i + 1, #mat.sub_dims do
                small_stride = small_stride * mat.sub_dims[j]
            end
            table.insert(mat.strides, {big_stride, small_stride * mat.td_size, mat.sub_dims[i]})
        end
    end


    -- required_grad
    if params.required_grad == false then
        mat.required_grad = false
    end

    if mat.required_grad == true then
        for i = 1, mat.size do
            mat.grad[i] = 0
        end
    end
    
    return mat
end 

function matrix.initialise_matrix(matrix, type)
    if type == "xavier" then
        for i = 1, matrix.size do
            matrix.data[i] = utils.random.xavierUniform(matrix.dims[1], matrix.dims[2])
        end
    end
end

function matrix:view(dimensions)
    local total_size = 1
    for i = 1, #dimensions do
        total_size = total_size * dimensions[i]
    end
    if total_size ~= self.size then
        error_handling.show_error("Invalid dimensions for matrix.")
    end

    -- copy dims
    self.dims[1] = dimensions[#dimensions - 1]
    self.dims[2] = dimensions[#dimensions]
    if #dimensions > 2 then
        for i = 1, #dimensions - 2 do
            table.insert(self.sub_dims, dimensions[i])
        end
    end

    -- set strides and size
    self.td_size = self.dims[1] * self.dims[2]
    if #self.sub_dims > 0 then
        for i = 1, #self.sub_dims do
            local big_stride = 1
            for j = 1, i do
                big_stride = big_stride * self.sub_dims[j]
            end
            local small_stride = 1
            for j = i + 1, #self.sub_dims do
                small_stride = small_stride * self.sub_dims[j]
            end
            table.insert(self.strides, {big_stride, small_stride * self.td_size, self.sub_dims[i]})
        end
    end
end

function matrix:shape()
    local dimensions = {}

    for i = 1, #self.sub_dims do
        table.insert(dimensions, self.sub_dims[i])
    end
    table.insert(dimensions, self.dims[1])
    table.insert(dimensions, self.dims[2])
    return dimensions
end

function matrix:print(params)
    if params == nil then
        params = {}
    end

    local function print_dim(table, dim_index, stride)
        if dim_index <= #self.sub_dims and self.sub_dims[dim_index] ~= nil then
            for i = 0, self.sub_dims[dim_index] - 1 do
                for j = 1, dim_index - 1 do 
                    io.write("  ")
                end
                io.write("{\n")
                print_dim(table, dim_index + 1, stride + self.strides[dim_index][2] * i)
                for j = 1, dim_index - 1 do 
                    io.write("  ")
                end
                io.write("}\n")
            end
        else
            for i = 0, self.dims[1] - 1 do
                for j = 1, dim_index - 1 do 
                    io.write("  ")
                end
                io.write("{ ")
                for j = 0, self.dims[2] - 1 do
                    io.write(string.format("%.4f", table[i * self.dims[2] + j + stride + 1]))
                    if j ~= self.dims[2] - 1 then
                        io.write(", ")
                    end
                end
                io.write(" }")
                if i ~= self.dims[1] then
                    io.write("\n")
                end
            end 
        end
    end

    -- print data
    if params.data ~= false then
        io.write("Data = ")
        print_dim(self.data, 1, 0)
    end

    -- print shape
    if params.shape == true then
        io.write("Shape: [")
        if #self.sub_dims ~= 0 then
            for i = 1, #self.sub_dims do
                io.write(self.sub_dims[i])
                io.write(", ")
            end
        end

        for i = 1, #self.dims do
            io.write(self.dims[i])
            if i ~= #self.dims then
                io.write(", ")
            end
        end
        io.write("]\n")
    end

    -- print grad
    if params.grad == true then
        io.write("Grad = ")
        print_dim(self.grad, 1, 0)
    end

    if params.strides == true and #self.strides ~= 0 then
        io.write("Strides = ")
        for i = 1, #self.strides do
            io.write("{")
            io.write(self.strides[i][1])
            io.write(", ")
            io.write(self.strides[i][2])
            io.write(", ")
            io.write(self.strides[i][3])
            io.write("}")
            if i ~= #self.strides then
                io.write(", ")
            end
        end
        io.write("\n")
    end

end

function matrix:copy(params)
    if params == nil then
        params = {}
    end

    local dimensions = {}
    local mat_params = {dims = dimensions}

    if params.data ~= nil then
        mat_params.data = params.data
    end
    if params.required_grad ~= nil then
        mat_params.required_grad = params.required_grad
    end

    for i = 1, #self.sub_dims do
        table.insert(dimensions, self.sub_dims[i])
    end
    table.insert(dimensions, self.dims[1])
    table.insert(dimensions, self.dims[2])

    local result = matrix.new(mat_params)

    if params.data == nil then
        for i = 1, self.size do
            result.data[i] = self.data[i]
        end
    end

    return result
end

return matrix

