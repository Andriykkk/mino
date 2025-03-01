-- package.path = package.path .. ";./mino/?.lua;./utils/datasets/?.lua;./utils/?.lua;"
-- local mino = require('mino')
-- local utils = require('utils')
-- local matrix = mino.Matrix

-- Define the function
local my_function = function(x, y)
    return x + y
end

-- Save the function as bytecode (use binary mode)
local file = io.open("my_function.lua", "wb")
file:write(string.dump(my_function))  -- Remove "return " prefix
file:close()

-- Load the function from bytecode
local file = io.open("my_function.lua", "rb")
local bytecode = file:read("*a")
file:close()

-- Load the function directly
local loaded_function = load(bytecode)

-- Use normally with single call
local result = loaded_function(2, 3)
print(result)  -- Output: 5
