local error_handling = {}

error_handling.message = "something went wrong"

function error_handling.show_error(message)
    print(message .. "\n" .. debug.traceback())
    os.exit(1)
end

return error_handling