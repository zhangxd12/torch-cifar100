require 'torch'
require 'paths'
require 'image'

local url = 'http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'
local tar = 'cifar-100-binary.tar.gz'
local extDir = 'cifar-100-binary'
local targetDir = 'cifar-100-batches.t7'

-- Download binary data
if not paths.filep(tar) then
    os.execute('wget -c ' .. url)
    os.execute('tar xvf ' .. paths.basename(url))
end

local function process(input, output)
    local m = torch.DiskFile(input, 'r'):binary()

    -- Seek to end to get the file size
    m:seekEnd()
    local length = m:position() - 1
    local nSamples = length / (32 * 32 * 3 + 2)

    assert(nSamples == math.floor(nSamples),
           'nSamples should be an exact integer!')

    -- Seek to first position to read data
    m:seek(1)
    local coarse = torch.ByteTensor(nSamples)
    local fine = torch.ByteTensor(nSamples)
    local data = torch.ByteTensor(nSamples, 3, 32, 32)
    for i = 1,nSamples do
        coarse[i] = m:readByte()
        fine[i] = m:readByte()
        local temp = m:readByte(32 * 32 * 3)
        data[i]:copy(torch.ByteTensor(temp))
    end

    local out = {}
    out.data = data
    out.labels = fine
    out.labelsCoarse = coarse
    torch.save(output, out)
end

os.execute('mkdir -p ' .. targetDir)
process(paths.concat(extDir, 'train.bin'),
        paths.concat(targetDir, 'data_batch.t7'))
process(paths.concat(extDir, 'test.bin'),
        paths.concat(targetDir, 'test_batch.t7'))

-- Delete downloaded file
os.execute('rm ' .. tar)
os.execute('rm -rf ' .. extDir)
