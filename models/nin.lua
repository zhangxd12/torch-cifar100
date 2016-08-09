require 'nn'

local model = nn.Sequential()

local function Block(...)
    local args = {...}
    model:add(nn.SpatialConvolution(...))
    model:add(nn.SpatialBatchNormalization(args[2], 1.0e-3))
    model:add(nn.ReLU(true))
    return model
end

Block(3, 192, 5, 5, 1, 1, 2, 2)
Block(192, 160, 1, 1)
Block(160, 96, 1, 1)
model:add(nn.SpatialMaxPooling(3, 3, 2, 2):ceil())
model:add(nn.Dropout())

Block(96, 192, 5, 5, 1, 1, 2, 2)
Block(192, 192, 1, 1)
Block(192, 192, 1, 1)
model:add(nn.SpatialAveragePooling(3, 3, 2, 2):ceil())
model:add(nn.Dropout())

Block(192, 192, 3, 3, 1, 1, 1, 1)
Block(192, 192, 1, 1)
Block(192, 10, 1, 1)
model:add(nn.SpatialAveragePooling(8, 8, 1, 1):ceil())
model:add(nn.View(10))

model:add(nn.LogSoftMax())

return model
