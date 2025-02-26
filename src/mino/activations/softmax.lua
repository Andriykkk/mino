local matrix = require('matrix')
local softmax = {}

-- UTILS
local function sub_dims_divider_single(dim_index, self_s, func, self)
    if dim_index <= #self.sub_dims then
        for i = 0, self.sub_dims[dim_index] - 1 do
            sub_dims_divider_single(dim_index + 1, self_s + self.strides[dim_index][2] * i, func, self)
        end
    else
        func(self, self_s)
    end
end
-- UTILS

function softmax.softmax(input)
    assert(input and input.dims, "Input must be a matrix with dimensions.")

    local result = input:copy()

    local function softmax_1d(self, self_s)
        for i = 0, self.dims[1] - 1 do
            local max_value = -math.huge
            for j = 0, self.dims[2] - 1 do
                if self.data[self_s + i * self.dims[2] + j + 1] > max_value then
                    max_value = self.data[self_s + i * self.dims[2] + j + 1]
                end
            end

            for j = 0, self.dims[2] - 1 do
                self.data[self_s + i * self.dims[2] + j + 1] = math.exp(self.data[self_s + i * self.dims[2] + j + 1] - max_value)
            end

            local sum_exp = 0
            for j = 0, self.dims[2] - 1 do
                sum_exp = sum_exp + self.data[self_s + i * self.dims[2] + j + 1]
            end
            for j = 0, self.dims[2] - 1 do
                self.data[self_s + i * self.dims[2] + j + 1] = self.data[self_s + i * self.dims[2] + j + 1] / sum_exp
            end
        end
    end

    sub_dims_divider_single(1, 0, softmax_1d, result)

    result.backward = softmax.softmax_backward
    result.operand1 = input
    
    return result
end

function softmax.softmax_backward(self, respect)
    local operand1 = self.operand1

    local result = operand1:copy({data = 0})

    local function softmax_1d_backward(self, self_s)
        for i = 0, self.dims[1] - 1 do
            local dot_product = 0
            for j = 0, self.dims[2] - 1 do
                local idx = self_s + i * self.dims[2] + j + 1
                dot_product = dot_product + self.data[idx] * respect.data[idx]
            end

            for j = 0, self.dims[2] - 1 do
                local idx = self_s + i * self.dims[2] + j + 1
                result.data[idx] = self.data[idx] * (respect.data[idx] - dot_product)
            end
        end
    end

    sub_dims_divider_single(1, 0, softmax_1d_backward, result)

    if operand1.required_grad == true then
        for i = 1, #result.data do
            operand1.grad[i] = operand1.grad[i] + result.data[i]
        end
    end

    if operand1.backward then
        operand1:backward(result)
    end
end

return softmax