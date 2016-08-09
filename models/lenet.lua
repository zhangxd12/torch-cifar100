require 'nn'

model = nn.Sequential()

model:add(nn.SpatialConvolution(3, 16, 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(nn.SpatialConvolution(16, 64, 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(nn.SpatialConvolution(64, 256, 5, 5))
model:add(nn.ReLU())

model:add(nn.View(256))
model:add(nn.Linear(256, 128))
model:add(nn.ReLU())
model:add(nn.Linear(128, 100))
model:add(nn.LogSoftMax())

return model
