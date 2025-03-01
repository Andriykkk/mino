local utils = {}

function utils.parameters_worker(self, func, params)
    for l_key, l_value in pairs(self.parameters) do
        if type(l_value) == "function" then
            goto continue
        end
        if l_value.optimiser_step ~= nil then
            l_value.optimiser_step(params)
            goto continue
        end
        if l_value.grad and l_value.data and l_value.required_grad then
            func(l_value, params)
            goto continue
        end
        if l_value.parameters ~= nil then
            for l_key_2, l_value_2 in pairs(l_value.parameters) do
                if type(l_value_2) ~= "function" and l_value_2.grad and l_value_2.data and l_value_2.required_grad then
                    func(l_value_2, params)
                end
            end
        end

        ::continue::
    end
end

function zero_gradients(matrix)
    for i = 1, matrix.size do
        matrix.grad[i] = 0
    end
end

function utils.zero_grad(self)
    for l_key, l_value in pairs(self.parameters) do
        if type(l_value) == "function" then
            goto continue
        end
        if l_value.optimiser_zero_grad ~= nil then
            l_value.optimiser_zero_grad()
            goto continue
        end
        if l_value.grad and l_value.data and l_value.required_grad then
            zero_gradients(l_value)
        end
        if l_value.parameters then
            for l_key_2, l_value_2 in pairs(l_value.parameters) do
                if type(l_value_2) ~= "function" and l_value_2.grad and l_value_2.data and l_value_2.required_grad then
                    zero_gradients(l_value_2)
                end
            end
        end

        ::continue::
    end
end

return utils