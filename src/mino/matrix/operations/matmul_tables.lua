local matmul_tables = {}

function matmul_tables.result(self, other, result, self_pos, other_pos, res_pos)
    local self_rows, self_cols = self.dims[1], self.dims[2]
    local other_rows, other_cols = other.dims[1], other.dims[2]

    for i = 0, self_rows - 1 do
        for j = 0, other_cols - 1 do
            local sum = 0
            for k = 0, self_cols - 1 do
                sum = sum + self.values[self_pos + i * self_cols + k + 1] * other.values[other_pos + k * other_cols + j + 1]
            end
            result.values[res_pos + i * other_cols + j + 1] = sum
        end
    end
end

return matmul_tables