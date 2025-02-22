local mul_tables = {}

function mul_tables.result(self, other, result, self_pos, other_pos, res_pos)
    for i = 0, result.dims[1] - 1 do
        for j = 0, result.dims[2] - 1 do
            local self_i = i % self.dims[1]
            local self_j = j % self.dims[2]

            local other_i = i % other.dims[1]
            local other_j = j % other.dims[2]

            local self_val = self.values[self_pos + self_i * self.dims[2] + self_j + 1]
            local other_val = other.values[other_pos + other_i * other.dims[2] + other_j + 1]

            result.values[res_pos + i * result.dims[2] + j + 1] = self_val * other_val
        end
    end
end

function mul_tables.back(self, other, result, self_pos, other_pos, res_pos)
    for i = 0, other.dims[1] - 1 do
        for j = 0, other.dims[2] - 1 do
            local self_i = i % self.dims[1]
            local self_j = j % self.dims[2]

            local result_i = i % result.dims[1]
            local result_j = j % result.dims[2]

            local self_val = self.values[self_pos + self_i * self.dims[2] + self_j + 1]
            local other_val = other.values[other_pos + i * other.dims[2] + j + 1]

            result.values[res_pos + result_i * result.dims[2] + result_j + 1] = result.values[res_pos + result_i * result.dims[2] + result_j + 1] + self_val * other_val 
        end
    end
end

return mul_tables