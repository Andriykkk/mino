local error_handling = require('error_handling')
local matrix = {}

-- UTILS
local function max(a, b)
    if a > b then
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
    if not(self.dims[1] ~= other.dims[1] or (self.dims[1] ~= 1 or other.dims[1] ~= 1)) then
        error_handling.dimension_error(self, other)
    elseif not(self.dims[2] ~= other.dims[2] or (self.dims[2] ~= 1 or other.dims[2] ~= 1)) then
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
    elseif self.dims[1] ~= other.dims[2] then
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
-- UTILS

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
            return result
        end

        sub_dims_divider(1, 0, 0, 0, add_tables, self, other, result)

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
            return result
        end

        sub_dims_divider(1, 0, 0, 0, subtract_tables, self, other, result)

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

        return result
    end,
    __mul = function(self, other)
        -- UTILS
        local function multiply_tables(self, other, result, self_pos, other_pos, res_pos)
            for i = 0, result.dims[1] - 1 do
                for j = 0, result.dims[2] - 1 do
                    local self_i = i % self.dims[1]
                    local self_j = j % self.dims[2]

                    local other_i = i % other.dims[1]
                    local other_j = j % other.dims[2]

                    local self_val = self.data[self_pos + self_i * self.dims[2] + self_j + 1]
                    local other_val = other.data[other_pos + other_i * other.dims[2] + other_j + 1]

                    result.data[res_pos + i * result.dims[2] + j + 1] = self_val * other_val
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
                result.data[i] = self.data[i] * other
            end
            return result
        end

        sub_dims_divider(1, 0, 0, 0, multiply_tables, self, other, result)

        return result
    end
}

function matrix:matmul(other)
    -- UTILS
    local function matmul_tables(self, other, result, self_pos, other_pos, res_pos)
        local a_rows, a_cols = a.dims[1], a.dims[2]
        local b_rows, b_cols = b.dims[1], b.dims[2]

        for i = 0, a_rows - 1 do
            for j = 0, b_cols - 1 do
                local sum = 0
                for k = 0, a_cols - 1 do
                    sum = sum + self.data[self_pos + i * a_cols + k + 1] * other.data[other_pos + k * b_cols + j + 1]
                end
                result.data[res_pos + i * b_cols + j + 1] = sum
                print(res_pos + i * b_cols + j + 1, sum)
            end
        end
    end
    -- UTILS
    if type(other) == "number" then
        error_handling.show_error("Can't multiply matrix with number.")
    end

    local result = create_result_matmul(self, other)

    sub_dims_divider(1, 0, 0, 0, matmul_tables, self, other, result)

    return result
end

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
        elseif type(params.fill_value) == "ones" then
            fill_table(mat.data, mat.size, 1)
        elseif type(params.fill_value) == "zeros" then
            fill_table(mat.data, mat.size, 0)
        elseif type(params.fill_value) == "random" then
            for i = 1, mat.size do
                mat.data[i] = math.random()
            end
        elseif type(params.fill_value) == "normal" then
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

    if params.strides == true then
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

return matrix

