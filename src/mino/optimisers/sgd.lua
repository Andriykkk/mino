local sdg = {}

function apply_gradients(matrix, lr)
    for i = 1, matrix.size do
        matrix.data[i] = matrix.data[i] - lr * matrix.grad[i]
    end
end

function zero_gradients(matrix)
    for i = 1, matrix.size do
        matrix.grad[i] = 0
    end
end

function print_keys(table)
    local keys = ""
    for key, value in pairs(table) do
        keys = keys .. key .. " "
    end
    print(keys)
end

function sdg.new(params)
    assert(params ~= nil, "Params cannot be nil in sgd")
    assert(params.parameters.parameters ~= nil, "Parameters cannot be nil in sgd")

    if params.learning_rate == nil then
        print("Learning rate not specified, defaulting to 0.01")
        params.learning_rate = 0.01
    end

    local optimiser = { parameters = params.parameters.parameters, learning_rate = params.learning_rate }

    function optimiser:step()
        for l_key, l_value in pairs(self.parameters) do
            if type(l_value) == "function" then
                goto continue
            end
            if l_value.grad and l_value.data and l_value.required_grad then
                apply_gradients(l_value, self.learning_rate)
            end
            if l_value.parameters then
                for l_key_2, l_value_2 in pairs(l_value.parameters) do
                    if type(l_value_2) == "function" then
                        goto continue
                    end
                    if l_value_2.grad and l_value_2.data and l_value_2.required_grad then
                        apply_gradients(l_value_2, self.learning_rate)
                    end
                end
            end

            ::continue::
        end
    end

    function optimiser:zero_grad()
        for l_key, l_value in pairs(self.parameters) do
            if type(l_value) == "function" then
                goto continue
            end
            if l_value.grad and l_value.data and l_value.required_grad then
                zero_gradients(l_value)
            end
            if l_value.parameters then
                for l_key_2, l_value_2 in pairs(l_value.parameters) do
                    if type(l_value_2) == "function" then
                        goto continue
                    end
                    if l_value_2.grad and l_value_2.data and l_value_2.required_grad then
                        zero_gradients(l_value_2)
                    end
                end
            end

            ::continue::
        end
    end

    return optimiser
end

return sdg