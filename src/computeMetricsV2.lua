-- Load global libraries
require("nn")
require("optim")
require("xlua") 

torch.setdefaulttensortype('torch.FloatTensor') 

require("nnsparse")

dofile("tools/CFNTools.lua")
dofile("tools/Appender.lua")
dofile("misc/Preload.lua")

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
cmd:text()


unpack = table.unpack
--the following code was not clean... sorry for that!



local ratioStep = 0.2
local params = cmd:parse(arg)

print("Options: ")
for key, val in pairs(params) do
  print(" - " .. key  .. "  \t : " .. tostring(val))
end

local ratioStep = params.ratioStep

params.loadFull = true
--Load data
print("Loading data...")
local train, test, info, matrixSize, lookup = LoadData(params.file, params)

targetType = "U"
if params.type == "U" then 
  targetType = "V"
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



-- Configure prediction error
local rmseFct = nnsparse.SparseCriterion(nn.MSECriterion())
local maeFct  = nnsparse.SparseCriterion(nn.AbsCriterion())

rmseFct.sizeAverage = false
maeFct.sizeAverage  = false

local batchSize = 20 
local curRatio  = ratioStep
local rmse, mae = 0,0

local f1Threshold = 0.0
-- local f1Ns = {5,10,15,20,30}
local f1Ns = {20}
local f1Info = {}
for i = 1, #f1Ns do
  f1Info[f1Ns[i]] = {tp=0,fp=0,tn=0,fn=0,misses=0}
end




--this method compute the error with the sparse matrix 
local transposeError = {}
function computeTranspose(outputs, targets, reverseIndex)
   for cursor, oneTarget in pairs(targets) do
       local i = reverseIndex[cursor]
   
       for k = 1, oneTarget:size(1) do
   
         local j = oneTarget[k][1]
   
         local y = outputs[cursor][j]
         local t = oneTarget[k][2]
   
         local mse = ( y - t )^2
   
         local transpose = transposeError[j] or nnsparse.DynamicSparseTensor(500)
         transpose:append(torch.Tensor{i, mse})
         transposeError[j] = transpose
   
       end
     end
end

function computeTranposeRatio(transposeError)

   --Sort samples by number of ratings
   local noRatings = nnsparse.DynamicSparseTensor(10000)
   local size  = 0
   for k, oneTranspose in pairs(transposeError) do
     transposeError[k] = oneTranspose:build():ssortByIndex()
     oneTranspose  = transposeError[k]
     size = size + 1
     noRatings:append(torch.Tensor{k, oneTranspose:size(1)})
   end
   
   noRatings = noRatings:build():ssort()
   local index = noRatings[{{},1}]
   
   
   local ignore = 0
   for kk = 1, size do
      if transposeError[index[kk]] == nil then ignore = ignore + 1 end
   end
   
   print("TRANSPOSE !!!")
   
   local curRatio = ratioStep
   local rmse   = 0
   local rmseInterval = 0
   local noSample = 0
   local noSampleInterval = 0
   
   for kk = 1, index:size(1) do
      local k    = index[kk]
      local data = transposeError[k][{{}, 2}]
      
      noSample         = noSample         + data:size(1)
      noSampleInterval = noSampleInterval + data:size(1)
      
      rmse         = rmse         + data:sum()
      rmseInterval = rmseInterval + data:sum()
      
      if kk >= curRatio * (size-ignore) then
           local curRmse = math.sqrt(rmse/noSample)*2
           rmseInterval  = math.sqrt(rmseInterval/noSampleInterval)*2
           print( kk .."/" ..  (size-ignore)  .. "\t ratio [".. curRatio .."] : " .. curRmse .. "\t Interval [".. (curRatio - ratioStep) .. "-".. curRatio .. "]: " .. rmseInterval)
           curRatio = curRatio + ratioStep 
           rmseInterval = 0
           noSampleInterval = 0
      end    
   end
   
   rmse = math.sqrt(rmse/noSample) * 2 
   
   print("Final RMSE: " .. rmse)

end

local numUserHitsCalculated = 0

function calculateHits(nTargets, itemScores, targetItems, N)
  numUserHitsCalculated = numUserHitsCalculated + 1
  -- print("calculateHits called")
  local sumA, sumB, sumAB = 0, 0, 0
  local misses = 0
  if itemScores:size(1) < N then misses = N - itemScores:size(1) end
  for i = 1, itemScores:size(1) do
    if i <= N then
      sumA = sumA + 1
      if targetItems[itemScores[i][1]] ~= nil then
        sumAB = sumAB + 1
      end 
    else
      break
    end
  end
  for k, v in pairs(targetItems) do
    sumB = sumB + 1
  end

  -- Should do: if sumB > N then sumB = N end

  local tp = sumAB
  local fp = sumA - sumAB
  local fn = sumB - sumAB
  local tn = nTargets - fn - fp - tp 
  return tp, fp, fn, tn, misses
end


local i = 1
local noSample = 0
local rmseInterval = 0
local noSampleInterval = 0

--Prepare minibatch
local inputs  = {}
local targets = {}
local hitTargets = {} 
local hiddens = {}

-- prepare meta-data
local denseMetadata  = train[1].new(batchSize, info.metaDim or 0)
local sparseMetadata = {}


------------MAIN!!!
local reverseIndex = {}
local usersSkipped = 0

local targetUser = 2
local targetK = lookup.U[targetUser]
local targetUserIndex = -1
local tuMean = 0

--for k, input in pairs(train) do
for kk = 1, size do
  local k = sortedIndex[kk]

  if kk == targetK then
    targetUserIndex = i
    tuMean = info[k].mean
  end
  -- Focus on the prediction aspect
  local input  = train[k]
  local target = test[k]
  local hidden = {}
  for i = 1, input:size(1) do
    hidden[input[i][1]] = true
  end

  -- Ignore data with no testing examples
  if target ~= nil then

    -- keep the original index
    reverseIndex[i] = k

    
    inputs[i]  = input
    hiddens[i] = hidden
    targets[i] = targets[i] or target.new()
    targets[i]:resizeAs(target):copy(target)

    hitTargets[i] = {}
    for ti = 1, targets[i]:size(1) do
      --print(targets[i])
      if targets[i][ti][2] >= f1Threshold then
        hitTargets[i][targets[i][ti][1]] = true
      end
    end

    -- center the target values
    targets[i][{{}, 2}]:add(-info[k].mean)

    if appenderIn then
      denseMetadata[i]  = info[k].full
      sparseMetadata[i] = info[k].fullSparse
    end
    
    noSample         = noSample         + target:size(1)
    noSampleInterval = noSampleInterval + target:size(1)
    --i = i + 1



    --compute loss when minibatch is ready
    if #inputs == batchSize then

      --Prepare metadata
      if appenderIn then
        appenderIn:prepareInput(denseMetadata, sparseMetadata)
      end

      local outputs = network:forward(inputs)

      for ui = 1, outputs:size(1) do
        local urecs = nnsparse.DynamicSparseTensor(10000)
        for ii = 1, outputs[ui]:size(1) do
          itemid = ii
          if hiddens[ui][itemid] == nil then
            urecs:append(torch.Tensor{itemid, outputs[ui][ii]})
          end
        end
        urecs = urecs:build():ssort(true)
        if ui == targetUserIndex then
          -- recs
          print( ui .."recs: ")
          for i = 1, urecs:size(1) do
            if i <= 20 then
              print("item:" .. lookup.IID[urecs[i][1]] .. " score:" .. urecs[i][2])
            else
              break
            end
          end
          -- inputs
           print("inputs: " .. input:size(2))
          for i = 1, input:size(1) do
            print("item:" .. lookup.IID[input[i][1]] .. " score:" .. input[i][2])
          end
          -- outputs
          print("outputs: " .. target:size(1))
          for i = 1, target:size(1) do
            print("item:" .. lookup.IID[target[i][1]] .. " score:" .. target[i][2])
          end
          print("user mean: " .. tuMean)
          targetUserIndex = -1
        end
        for fi = 1, #f1Ns do
          local f1n = f1Ns[fi]
	  -- print(urecs)
          local tp, fp, fn, tn, misses = calculateHits(matrixSize[targetType], urecs, hitTargets[ui], f1n)
          f1Info[f1n].tp = f1Info[f1n].tp + tp
          f1Info[f1n].fp = f1Info[f1n].fp + fp
          f1Info[f1n].fn = f1Info[f1n].fn + fn
          f1Info[f1n].tn = f1Info[f1n].tn + tn
          f1Info[f1n].misses = f1Info[f1n].misses + misses
        end
      end

      -- compute MAE
      mae = mae + maeFct:forward(outputs, targets)

      --compute RMSE
      local rmseCur = rmseFct:forward(outputs, targets)
      rmse         = rmse        +  rmseCur
      rmseInterval = rmseInterval + rmseCur

      --reset minibatch
      inputs = {}
      hitTargets = {}
      hiddens = {}
      i = 1
      
      -- if the ratio
      if kk >= curRatio * (size-ignore) then
      
        local curRmse = math.sqrt(rmse/noSample)*2
        rmseInterval  = math.sqrt(rmseInterval/noSampleInterval)*2
        
        print( kk .."/" ..  (size-ignore)  .. "\t ratio [".. curRatio .."] : " .. curRmse .. "\t Interval [".. (curRatio - ratioStep) .. "-".. curRatio .. "]: " .. rmseInterval)
        
        -- increment next ratio
        curRatio = curRatio + ratioStep
        
        -- reset interval
        rmseInterval     = 0
        noSampleInterval = 0
      end
      
      
      computeTranspose(outputs, targets, reverseIndex)
      reverseIndex = {}

    else
      i = i + 1
    end

  else
    usersSkipped = usersSkipped + 1
  end
end

-- remaining data for minibatch
if #inputs > 0 then
  local _targets = {unpack(targets, 1, #inputs)} --retrieve a subset of targets

  if appenderIn then
    local _sparseMetadata = {unpack(sparseMetadata, 1, #inputs)}
    local _denseMetadata =  denseMetadata[{{1, #inputs},{}}]

    appenderIn:prepareInput(_denseMetadata, _sparseMetadata)
  end

  local outputs = network:forward(inputs)

  for ui = 1, outputs:size(1) do
    local urecs = nnsparse.DynamicSparseTensor(10000)
    for ii = 1, outputs[ui]:size(1) do
      itemid = ii
      if hiddens[ui][itemid] == nil then
        urecs:append(torch.Tensor{itemid, outputs[ui][ii]})
      end
    end
    urecs = urecs:build():ssort(true)
    for fi = 1, #f1Ns do
      local f1n = f1Ns[fi]
      local tp, fp, fn, tn, misses = calculateHits(matrixSize[targetType], urecs, hitTargets[ui], f1n)
      f1Info[f1n].tp = f1Info[f1n].tp + tp
      f1Info[f1n].fp = f1Info[f1n].fp + fp
      f1Info[f1n].fn = f1Info[f1n].fn + fn
      f1Info[f1n].tn = f1Info[f1n].tn + tn
      f1Info[f1n].misses = f1Info[f1n].misses + misses
    end
  end

  mae  = mae  + maeFct:forward(outputs , _targets)


  local rmseCur = rmseFct:forward(outputs, _targets)
  rmse         = rmse        +  rmseCur
  rmseInterval = rmseInterval + rmseCur

  local curRmse  = math.sqrt(rmse/noSample )*2
  rmseInterval   = math.sqrt(rmseInterval/noSampleInterval)*2

  computeTranspose(outputs, _targets, reverseIndex)
  
  
end

rmse = math.sqrt(rmse/noSample) * 2 
mae  = mae/noSample * 2

print( (size-ignore) .."/" ..  (size-ignore)  .. "\t ratio [1.0] : " .. rmse .. "\t Interval [0.8-1.0]: " .. rmseInterval)

print("Final RMSE: " .. rmse)
print("Final MAE : " .. mae)

computeTranposeRatio(transposeError)

function fscore(tp, fp, fn, tn, beta)
  local p, r, f = 0.0,0.0,0.0
  if tp ~= 0 then
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    f = (1+beta*beta)*p*r/(beta*beta*p+r)
  end
  return p,r,f
end

function xlogx(x) 
  if x > 0 then
    return x*math.log(x)
  end
  return 0
end

function llscore(tp, fp, fn, tn)
  local N = tp + fp + fn + tn
  local x = xlogx(N) - xlogx(tp+fn) - xlogx(fp+tn)
  x = x + xlogx(tp) + xlogx(fp) - xlogx(tp+fp)
  x = x + xlogx(fn) + xlogx(tn) - xlogx(fn+tn)
  x = math.sqrt(x/N)
  if tp+fp == 0.0 or fn+tn == 0.0 then
    x = 0.0
  elseif tp/(tp+fp) < fn/(fn+tn) then
    x = -x
  end
  score = math.tanh(x)
  return x, score
end

function fllscore(tp, fp, fn, tn, misses, beta)
  local p,r,f = fscore(tp, fp, fn, tn, beta)
  local ll, lltanh = llscore(tp, fp, fn, tn)
  print("True Positives:\t\t" .. tp .."\nFalse Positives:\t" .. fp .. "\nFalse Negatives:\t" .. fn .. "\nTrue Negatives:\t" .. tn .. "\nMisses:\t" .. misses)

  print("\nF" .. beta .. "-Score\n--------\n")
  print("Precision:\t" .. p .. "\nRecall:\t\t" .. r .. "\nF" .. beta .. "-Score:\t" .. f .. "\n")

  print("Log-Likelihood\n--------------\n")
  print("Log-Likelihood Score:\t" .. ll .. "\nTanh(llscore/sqrt(N)):\t" .. lltanh .. "\n\n\n")
end

print("numUserHitsCalculated: " .. numUserHitsCalculated .. "\nnumUsersSkipped:" .. usersSkipped)
for k, v in pairs(f1Info) do
  print("Accuracy Measure N:" .. k .. " -------------------")
  fllscore(v.tp, v.fp, v.fn, v.tn, v.misses, 1)
end


 
