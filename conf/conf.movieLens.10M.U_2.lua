config = 
{
   useMetadata = false,
   layer1 = 
   {      
      layerSize = 800,
      { 
         criterion = cfn.SDAECriterionGPU(nn.MSECriterion(),
         {
            alpha = 1.0336864752695,
            beta  = 0.38166233734228,
            hideRatio = 0.25525745830964,
         }), 
         noEpoch = 80, 
         miniBatchSize = 35,
         learningRate = 0.061005359655246,  
         learningRateDecay = 0.54645854830742,
         weightDecay = 0.00570531450212,
      },
      
   },
   layer2 = 
   {
      layerSize = 500,
      { 
         criterion = cfn.SDAECriterionGPU(nn.MSECriterion(),
         {
            alpha = 1,
            beta  = 0.8,
            noiseRatio = 0.2,
            noiseStd  = 0.02, 
         }),
         noEpoch = 10, 
         miniBatchSize = 5,
         learningRate  = 1e-5,  
         learningRateDecay = 0.1,
         weightDecay = 0.2,
         momentum = 0.8
      },
      
      {
         criterion = cfn.SDAECriterionGPU(nn.MSECriterion(),
         {
            alpha = 0.98560892462868,
            beta  = 0.58072139311116,
            hideRatio = 0.12767389068742,
         }),
         noEpoch = 20,
         miniBatchSize = 25,
         learningRate  = 0.010696778637008,
         learningRateDecay = 0.45096496415014,
         weightDecay = 0.015386821936839,
         
      },
      
   },
}

return config
