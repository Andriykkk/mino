local loss = {}
local cross_entropy = require('loss/cross_entropy')

function loss.cross_entropy(input, target)
    return cross_entropy.cross_entropy(input, target)
end

return loss