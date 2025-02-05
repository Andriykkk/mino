local multiplication_kernel = {}

function multiplication_kernel.run_big(a, b, result, a2, b2, s1, s2, s3, s4)
    for i = 1, s1 do
        for ii = 1, s2 do
            for j = 1, s3 do
                for jj = 1, s4 do
                    local row = (i - 1) * s2 + ii
                    local col = (j - 1) * s4 + jj

                    result[(row - 1) * a2 + col] = a[(row - 1) * a2 + col] * b[(ii - 1) * b2 + jj]
                end
            end
        end
    end
end

function multiplication_kernel.run_small(a, b, result, a2, b2, s1, s2, s3, s4)
    for i = 1, s1 do
        for ii = 1, s2 do
            for j = 1, s3 do
                for jj = 1, s4 do
                    local row = (i - 1) * s2 + ii
                    local col = (j - 1) * s4 + jj
                    
                    result[(ii - 1) * b2 + jj] = result[(ii - 1) * b2 + jj] + a[(row - 1) * a2 + col] * b[(ii - 1) * b2 + jj]
                end
            end
        end
    end
end

function multiplication_kernel.run_big_back(a, b, a2, b2, s1, s2, s3, s4)
    for i = 1, s1 do
        for ii = 1, s2 do
            for j = 1, s3 do
                for jj = 1, s4 do
                    local row = (i - 1) * s2 + ii
                    local col = (j - 1) * s4 + jj

                    a[(row - 1) * a2 + col] = a[(row - 1) * a2 + col] * b[(ii - 1) * b2 + jj]
                end
            end
        end
    end
end

return multiplication_kernel