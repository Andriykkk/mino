package.path = package.path .. ";./mino/?.lua;./utils/datasets/?.lua;./utils/?.lua;"
local progress_bar = require('progress_bar')

local total_steps = 100000
local bar = progress_bar:new(total_steps)

for i = 1, total_steps do
    bar:step({learning_rate = 1.12, momentum = 0.9})
    local t = os.clock()
    while os.clock() - t < 0.001 do end
end

print()