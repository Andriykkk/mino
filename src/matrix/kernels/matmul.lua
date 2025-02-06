local matmul_kernel = {}

function matmul_kernel.matmul_naive(self, other, result)
    for i = 1, self.dims[1] do
        for j = 1, other.dims[2] do
            local sum = 0
            for k = 1, self.dims[2] do
                sum = sum + self.data[(i - 1) * self.dims[2] + k] * other.data[(k - 1) * other.dims[2] + j]
            end
            result.data[(i - 1) * other.dims[2] + j] = sum
        end
    end
end

return matmul_kernel