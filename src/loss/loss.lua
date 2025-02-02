local loss = {}
local categoricalcrossentropy = require('loss/categoricalcrossentropy')

function loss.cross_entropy(input, target)
    return categoricalcrossentropy.categoricalcrossentropy(input, target)
end

return loss