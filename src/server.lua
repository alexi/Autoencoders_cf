-- Load global libraries
require("nn")
require("optim")
require("xlua") 

torch.setdefaulttensortype('torch.FloatTensor') 

require("nnsparse")

dofile("tools/CFNTools.lua")
dofile("tools/Appender.lua")
dofile("misc/Preload.lua")

-- 1. Lookup Table
-- 2. Load nn
-- 3. For user, get recs OR accept item-ratings vector as input to feed through nn?
-- 4. Sort recs, pick top N

----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Compute final metrics for network')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-file'              , ''  ,  'The relative path to your data file (torch format)')
cmd:option('-network'           , ""  , 'The relative path to the lua configuration file')
cmd:option('-type'              , ""  , 'The network type U/V')
cmd:option('-gpu'               , 1   , 'use gpu')
cmd:option('-ratioStep'         , 0.2   , 'use gpu')
cmd:option('-numericIds'        , false  , 'Items and Users use numeric ids')
cmd:text()



--the following code was not clean... sorry for that!



local ratioStep = 0.2
local params = cmd:parse(arg)

print("Options: ")
for key, val in pairs(params) do
  print(" - " .. key  .. "  \t : " .. tostring(val))
end

local ratioStep = params.ratioStep

--Load lookup tables
params.loadFull = true

--Load data
print("Loading data...")
local train, test, info, matrixSize, lookup = LoadData(params.file, params)

--Seed target type index:name table
targetType = "U"
if params.type == "U" then 
  targetType = "V"
end

targetTypeIndexToId = {}
for k, v in pairs(lookup[params.type]) do
  targetTypeIndexToId[v[1]] = k
end

-- start evaluating
print("Loading network...")
local network = torch.load(params.network)
network:evaluate()
print(network)


--look for appenderIn
local appenderIn
for k = 1, network:size() do
  local layer = network:get(k)
  if torch.type(layer) == "cfn.AppenderOut" then
    appenderIn = layer.appenderIn
    print("AppenderIn found")
    break
  end
end






--Sort samples by their number of ratings
local noRatings = nnsparse.DynamicSparseTensor(10000)
local size = 0
for k, oneTrain in pairs(train) do
  size = size + 1
  noRatings:append(torch.Tensor{k, oneTrain:size(1)})
end
noRatings = noRatings:build():ssort()
local sortedIndex = noRatings[{{},1}]


-- compute the number of valid training samples
local ignore = 0
for kk = 1, size do
   if train[sortedIndex[kk]] == nil then ignore = ignore + 1 end
end

-- GET RECS
function getUserRecs(uid)
  local inputs  = {}
  local targets = {}
  -- prepare meta-data
  local denseMetadata  = train[1].new(1, info.metaDim or 0)
  local sparseMetadata = {}

  -- handle numeric ids
  if params.numericIds then uid = tonumber(uid) end

  local k = lookup.U[uid]
  local input  = train[k]
  local target = test[k]
  local hidden = {}
  for k, oneTrain in pairs(input) do
    for i = 1, oneTrain:size(1) do
      hidden[oneTrain[i][1]] = true
    end
  end

  if target ~= nil then  
    local i = 1  
    inputs[i] = input

    targets[i] = targets[i] or target.new()
    targets[i]:resizeAs(target):copy(target)

    -- center the target values
    targets[i][{{}, 2}]:add(-info[k].mean)

    if appenderIn then
      denseMetadata[i]  = info[k].full
      sparseMetadata[i] = info[k].fullSparse
    end

    --Prepare metadata
    if appenderIn then
      appenderIn:prepareInput(denseMetadata, sparseMetadata)
    end

    local outputs = network:forward(inputs)

    -- Recs gen
    local urecs = nnsparse.DynamicSparseTensor(10000)
    for ii = 1, outputs[ui]:size(1) do
      itemid = ii
      if hidden[itemid] ~= nil then
        urecs:append(torch.Tensor{itemid, outputs[ui][ii]})
        -- table.insert(urecs, {})
      end
    end
    urecs = urecs:build():ssort(true) -- sort high -> low rating val
    -- table.sort(urecs, function (left, right)
    --                     return left[2] > right[2]
    --                   end)
    return urecs
  end
end

