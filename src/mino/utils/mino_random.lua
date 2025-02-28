local mino_random = {}

function mino_random.xavierUniform(in_size, out_size)
    local range = math.sqrt(6 / (in_size + out_size))
    
    local random_value = math.random() * 2 * range - range
    
    return random_value
end

return mino_random