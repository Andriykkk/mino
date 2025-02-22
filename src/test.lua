-- Function to check speed of addition
function test_addition(n)
    local sum = 0
    for i = 1, n do
        sum = sum + i
    end
end

-- Function to check speed of modulo operation
function test_modulo(n)
    local mod_result = 0
    for i = 1, n do
        mod_result = i % 3
    end
end

-- Measure the time for addition operation
local start_time_add = os.clock()
test_addition(1000000000)  -- Running the addition loop 10 million times
local end_time_add = os.clock()
local time_taken_add = end_time_add - start_time_add
print("Time taken for addition: " .. time_taken_add .. " seconds")

-- Measure the time for modulo operation
local start_time_mod = os.clock()
test_modulo(1000000000)  -- Running the modulo loop 10 million times
local end_time_mod = os.clock()
local time_taken_mod = end_time_mod - start_time_mod
print("Time taken for modulo: " .. time_taken_mod .. " seconds")
