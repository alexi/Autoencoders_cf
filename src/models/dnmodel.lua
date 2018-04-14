function createModel(info, config)
	   -- retrieve layer size
   local metaDim = 0
   if config.use_meta then 
      metaDim = info.metaDim or 0
   end

   local bottleneck = {}
   bottleneck[0] = info.dimension
   local i = 1

   local confLayers = {}
   for key, confLayer in pairs(config) do
         if string.starts(key, "layer") then
            local lnum = tonumber(string.sub(key, 6))
            confLayers[lnum] = confLayer
            bottleneck[lnum] = confLayer.layerSize
         end
   end

   --Step 1 : Build networks
   local encoders = {}
   local decoders = {}
   local finalNetwork
   
    local appenderIn = nil
    if config.use_meta then
        appenderIn = cfn.AppenderIn:new()
    end
   
   
   local i = 0
   for i = 1, #confLayers do
      local confLayer = confLayers[i]
   
      --ENCODERS
      encoders[i] = nn.Sequential()
      
      if i == 1  then --sparse input
      
         if appenderIn then
            encoders[i]:add(cfn.AppenderSparseOut(appenderIn)) 
         end
         
          if config.use_gpu then
            encoders[i]:add(nnsparse.Densify(bottleneck[i-1] + metaDim))
            encoders[i]:add(      nn.Linear (bottleneck[i-1] + metaDim, bottleneck[i]))
         else
            encoders[i]:add(nnsparse.SparseLinearBatch(bottleneck[i-1] + metaDim, bottleneck[i], false))
         end

      else --dense input
      
         -- if appenderIn then 
         --    encoders[i]:add(cfn.AppenderOut(appenderIn)) 
         -- end
         
         -- encoders[i]:add(nn.Linear(bottleneck[i-1] + metaDim, bottleneck[i]))
         
         encoders[i]:add(nn.Linear(bottleneck[i-1], bottleneck[i]))
      end
               
      encoders[i]:add(nn.Tanh())
      
      --DECODERS
      decoders[i] = nn.Sequential()
      
      if i == 1 and appenderIn then 
         decoders[i]:add(cfn.AppenderOut(appenderIn)) 
         decoders[i]:add(nn.Linear(bottleneck[i] + metaDim ,bottleneck[i-1]))
      else
         decoders[i]:add(nn.Linear(bottleneck[i],bottleneck[i-1]))
      end

      decoders[i]:add(nn.Tanh())
      
      -- tied weights
      if confLayer.isTied == true then
         decoders[i]:get(1).weight     = encoders[i]:get(1).weight:t()
         decoders[i]:get(1).gradWeight = encoders[i]:get(1).gradWeight:t()
      end
   end
end