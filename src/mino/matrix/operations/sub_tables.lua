local sub_tables = {}

function sub_tables.one(self, other, self_pos, other_pos)
    for i = 0, other.dims[1] - 1 do
        for j = 0, other.dims[2] - 1 do
            local self_i = i % self.dims[1]
            local self_j = j % self.dims[2]

            local other_val = other.values[other_pos + i * other.dims[2] + j + 1]

            self.values[self_pos + self_i * self.dims[2] + self_j + 1] = self.values[self_pos + self_i * self.dims[2] + self_j + 1] - other_val
        end
    end
end

return sub_tables