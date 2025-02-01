local relu = {}

function relu.relu(input)
    for i = 1, #input.data do
        if input.data[i] < 0 then
            input.data[i] = 0
        end
    end 
end

return relu