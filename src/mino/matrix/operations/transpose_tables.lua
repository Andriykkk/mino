local transpose_tables = {}

-- UTILS
local function compute_strides(dims)
    local strides = {}
    local stride = 1
    for i = #dims, 1, -1 do
        strides[i] = stride
        stride = stride * dims[i]
    end
    return strides
end
-- UTILS

function transpose_tables.result(self, result, dim1, dim2, old_dims, new_dims)
    local orig_strides = compute_strides(old_dims)
    local transposed_strides = compute_strides(new_dims)

    for i = 1, #self.data do
        local indices = {}
        local remaining = i - 1
        for j = #old_dims, 1, -1 do
            indices[j] = remaining % old_dims[j] + 1
            remaining = math.floor(remaining / old_dims[j])
        end

        indices[dim1], indices[dim2] = indices[dim2], indices[dim1]

        local new_index = 1
        for j = 1, #new_dims do
            new_index = new_index + (indices[j] - 1) * transposed_strides[j]
        end

        result.data[new_index] = self.data[i]
    end

    return result
end

return transpose_tables