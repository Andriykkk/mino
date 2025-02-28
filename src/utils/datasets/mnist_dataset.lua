local mnist_dataset = {}

function mnist_dataset.read_csv(file_path, limit)
    local data = {}

    local file = io.open(file_path, "r")
    if not file then
        return nil
    end

    local index = 0
    for line in file:lines() do
        index = index + 1
        if limit ~= nil and index > limit then
            break
        end
        
        local row = {}
        for value in string.gmatch(line, "([^,]+)") do
            table.insert(row, tonumber(value))
        end

        local label = row[1]

        local pixels = {}
        for i = 2, #row do
            table.insert(pixels, row[i]/255.0)
        end

        table.insert(data, {label = label, pixels = pixels})
    end

    file:close()

    return data
end

function mnist_dataset.print_mnist(data, limit)
    for i = 1, limit do
        local row = data[i]
        local label = row.label
        local pixels = row.pixels

        print("Label: " .. label)

        for j = 1, 28 do
            for k = 1, 28 do
                local pixel = pixels[(j - 1) * 28 + k]
                if pixel > 0.7 then
                    io.write("#")
                elseif pixel > 0.3 then
                    io.write(".")
                else
                    io.write(" ")
                end
            end
            io.write("\n")
        end
    end
end

return mnist_dataset