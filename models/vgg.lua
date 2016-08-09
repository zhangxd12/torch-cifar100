require 'nn'

local model = nn.Sequential()

local function ConvBNReLU(nInput, nOutput)
    model:add(nn.SpatialConvolution(nInput, nOutput, 3, 3, 1, 1, 1, 1))
    model:add(nn.SpatialBatchNormalization(nOutput, 1.0e-3))
    model:add(nn.ReLU(true))
    return model
end

local MaxPooling = nn.SpatialMaxPooling

ConvBNReLU(3, 64)
model:add(nn.Dropout(0.3))
ConvBNReLU(64, 64)
model:add(MaxPooling(2, 2, 2, 2):ceil())

ConvBNReLU(64, 128)
model:add(nn.Dropout(0.4))
ConvBNReLU(128, 128)
model:add(MaxPooling(2, 2, 2, 2):ceil())

ConvBNReLU(128, 256)
model:add(nn.Dropout(0.4))
ConvBNReLU(256, 256)
model:add(nn.Dropout(0.4))
ConvBNReLU(256, 256)
model:add(MaxPooling(2, 2, 2, 2):ceil())

ConvBNReLU(256, 512)
model:add(nn.Dropout(0.4))
ConvBNReLU(512, 512)
model:add(nn.Dropout(0.4))
ConvBNReLU(512, 512)
model:add(MaxPooling(2, 2, 2, 2):ceil())

ConvBNReLU(512, 512)
model:add(nn.Dropout(0.4))
ConvBNReLU(512, 512)
model:add(nn.Dropout(0.4))
ConvBNReLU(512, 512)
model:add(MaxPooling(2, 2, 2, 2):ceil())

model:add(nn.View(512))

model:add(nn.Dropout(0.5))
model:add(nn.Linear(512, 512))
model:add(nn.BatchNormalization(512))
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(512, 100))

model:add(nn.LogSoftMax())

return model
