local error_handling = {}

error_handling.message = "something went wrong"

function error_handling.show_error(message)
    print(message .. "\n" .. debug.traceback())
    os.exit(1)
end

function error_handling.dimension_error(self, other)
    print("Matrices dimensions are not compatible. Trying to perform operation on matrix with shape " .. self.dims[1] .. "x" .. self.dims[2] .. " to matrix with shape " .. other.dims[1] .. "x" .. other.dims[2] .. "." .. "\n" .. debug.traceback())
    os.exit(1)
end

return error_handling