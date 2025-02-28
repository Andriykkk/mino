local error_handling = {}

function error_handling.show_error(message)
    print(message)
    os.exit(1)
end

function  error_handling.dimension_error(self, other)
    error_handling.show_error("Invalid dimensions for matrix operation for matrices " .. self.dims[1] .. "x" .. self.dims[2] .. " and " .. other.dims[1] .. "x" .. other.dims[2] .. ".")
end

return error_handling