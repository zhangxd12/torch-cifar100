require 'paths'
require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'dataset-cifar100'

opt = lapp[[
    -s,--save                  (default "logs")      subdirectory to save logs
    -b,--batchSize             (default 50)          batch size
    -r,--learningRate          (default 0.05)        learning rate
    --network                  (default nin)         network
    --learningRateDecay        (default 5.0e-7)      learning rate decay
    -m,--momentum              (default 0)           momentum
    --threads                  (default 4)           number of threads
    --max_epoch                (default 100)         maximum number of iterations
    --type                     (default float)       cuda/float
]]

-- Set # of threads
torch.setnumthreads(opt.threads)

-- Set floating point type
torch.setdefaulttensortype('torch.FloatTensor')

-- Class labels
-- classes = {}

-- Cast function
local function cast(t)
    if opt.type == 'cuda' then
        require 'cunn'
        return t:cuda()
    else
        return t:float()
    end
end

-- Define models
model = require('models/' .. opt.network)

print(model)

-- Criterion
criterion = nn.CrossEntropyCriterion()

-- Confusion matrix
confusion = optim.ConfusionMatrix(100)

-- Get dataset
nTrainingPatches = 5000
nTestingPatches = 1000

print('<cifar100> preparing training data')
trainData = cifar100.loadTrainSet(nTrainingPatches)
trainData:normalize(trainData)

print('<cifar100> preparing testing data')
testData = cifar100.loadTestSet(nTestingPatches)
testData:normalize(testData)

-- Loggers
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Cast model and criterion
model = cast(model)
criterion = cast(criterion)

-- Parameters
parameters, gradParameters = model:getParameters()

-- Training function
function train(dataset)
    model:training()

    epoch = epoch or 1

    local time = sys.clock()

    print('<trainer> epoch #' .. epoch)
    print('<trainer> batch size = ' .. (opt.batchSize))

    indices = torch.randperm(dataset:size())
    for t = 1,dataset:size(), opt.batchSize do
        -- Copy input data and labels
        local inputs = torch.Tensor(opt.batchSize, 3, 32, 32)
        local targets = torch.Tensor(opt.batchSize)
        local k = 1
        for i = t,math.min(t + opt.batchSize - 1, dataset:size()) do
            local input = dataset.data[indices[i]]:clone()
            inputs[k] = input
            targets[k] = dataset.labels[indices[i]]
            k = k + 1
        end

        -- Cast data/labels
        inputs = cast(inputs)
        targets = cast(targets)

        -- Evaluation function
        local feval = function(x)
            collectgarbage()

            if x ~= parameters then
                parameters:copy(x)
            end

            gradParameters:zero()

            local outputs = model:forward(inputs)
            local f = criterion:forward(outputs, targets)

            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)

            confusion:batchAdd(outputs, targets)

            return f, gradParameters
        end

        -- Stochastic gradient descent
        optimState = {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            learningRateDecay = opt.learningRateDecay
        }
        optim.sgd(feval, parameters, optimState)

        -- Show progress
        xlua.progress(t - 1, dataset:size())
    end

    -- Computation time
    time = sys.clock() - time
    time = time / dataset:size()
    print('<trainer> time to learn 1 sample = ' .. (time * 1000.0) .. ' ms')

    -- Confusion matrix
    confusion:updateValids()
    print(('Train accuracy: %10.6f'):format(100.0 * confusion.totalValid))
    confusion:zero()

    -- Save/log current network
    local filename = paths.concat(opt.save, 'cifar100.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<tester> saving network to ' .. filename)
    torch.save(filename, model)

    -- Increment epoch
    epoch = epoch + 1
end

function test(dataset)
    model:evaluate()

    local time = sys.clock()

    for t = 1,dataset:size(),opt.batchSize do
        -- Show progress
        xlua.progress(t-1, dataset:size())

        -- Copy test data/labels
        local inputs = torch.Tensor(opt.batchSize, 3, 32, 32)
        local targets = torch.Tensor(opt.batchSize)
        local k = 1
        for i = t,math.min(t+opt.batchSize-1, dataset:size()) do
            local input = dataset.data[i]:clone()
            inputs[k] = input
            targets[k] = dataset.labels[i]
            k = k + 1
        end

        -- Cast data/labels
        inputs = cast(inputs)
        targets = cast(targets)

        -- Predict
        local preds = model:forward(inputs)

        -- Update confusion matrix
        confusion:batchAdd(preds, targets)
    end

    -- Computaion time
    time = sys.clock() - time
    time = time / dataset:size()
    print('<tester> time to test 1 sample = ' .. (time * 1000.0) .. ' ms')

    -- Show confusion matrix
    confusion:updateValids()
    print(('Train accuracy: %6.2f'):format(100.0 * confusion.totalValid))
    confusion:zero()
end

----------------------------------------------------
-- Main process --
for i=1,opt.max_epoch do
    train(trainData)
    test(testData)
end
