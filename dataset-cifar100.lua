require 'torch'
require 'paths'

cifar100 = {}
cifar100.datasetDir = 'cifar-100-batches.t7'
cifar100.trainBatches = 'data_batch.t7'
cifar100.testBatches = 'test_batch.t7'

function cifar100.loadTrainSet(maxLoad)
    return cifar100.loadDataset(paths.concat(cifar100.datasetDir, cifar100.trainBatches), maxLoad)
end

function cifar100.loadTestSet(maxLoad)
    return cifar100.loadDataset(paths.concat(cifar100.datasetDir, cifar100.trainBatches), maxLoad)
end

function cifar100.loadDataset(filename, maxLoad)
    local f = torch.load(filename)

    local data = f.data:type('torch.FloatTensor')
    local labels = f.labels + 1

    local nExample = f.data:size(1)
    if maxLoad and maxLoad > 0 and maxLoad < nExample then
        nExample = maxLoad
    end
    print('<cifar100> load ' .. nExample .. ' datasets')

    local dataset = {}
    dataset.data = data
    dataset.labels = labels

    function dataset:normalize()
        data = self.data

        local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
        for i = 1,data:size(1) do
            xlua.progress(i, dataset:size())
            -- Convert to YUV
            local rgb = data[i]
            local yuv = image.rgb2yuv(rgb)
            -- Normalize Y channel
            yuv[1] = normalization(yuv[{{1}}])
            data[i] = yuv
        end

        -- Normalize U channel
        local mean_u = data:select(2, 2):mean()
        local std_u = data:select(2, 2):std()
        data:select(2, 2):add(-mean_u)
        data:select(2, 2):div(std_u)

        -- Normalize U channel
        local mean_u = data:select(2, 3):mean()
        local std_u = data:select(2, 3):std()
        data:select(2, 3):add(-mean_u)
        data:select(2, 3):div(std_u)
    end

    function dataset:size()
        return nExample
    end

    return dataset
end
