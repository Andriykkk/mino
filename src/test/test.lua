function test_matmul_speed(a, b)
    local startTime = os.clock()
    local c = mino.matmul(a, b)
    local endTime = os.clock()

    local elapsedTime = endTime - startTime
    return elapsedTime
end

function matrix_check()
    local file = io.open("matrix_check.txt", "w")

    if file then
        for i = 110, 500, 10 do
            print(i)
            for j = i - 100, i + 100, 10 do
                local a = mino.Matrix({dims = {i, j}, data = 2})
                local b = mino.Matrix({dims = {j, i}, data = 2})

                local time = test_matmul_speed(a, b)
                file:write(i * j .. " " .. time .. "\n")
            end
        end
        file:close()
    end

    local file = io.open("matrix_check2.txt", "w")

    if file then
        for i = 100, 500, 10 do
                local a = mino.Matrix({dims = {i, i}, data = 2})
                local b = mino.Matrix({dims = {i, i}, data = 2})

                local time = test_matmul_speed(a, b)
                file:write(i * i .. " " .. time .. "\n")
        end
        file:close()
    end
end
matrix_check()