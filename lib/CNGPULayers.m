(* ::Package:: *)

(* Don't forget to run CNGPUInitialize[] before attempting to use *)


(* Warning: Highly experimental code. Not well tested. Always cross check gradients against CPU versions. *)


Needs["CUDALink`"];


GPUCode="

__device__ void d_GPUConvolveFilterBankTo2D( float* input, float* filterBank, float* output, mint examples, mint filters, mint srcHeight, mint srcWidth, mint filterSize )
{
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;

   if ( x >= srcWidth-5+1 )
      return;
   if ( y >= srcHeight-5+1 )
      return;


   for ( int ex = 0 ; ex < examples ; ex++ )
   {
      float accum = 0.0;

      for ( int filt = 0 ; filt < filters ; filt++ )
         for ( int ty = 0 ; ty < 5 ; ty++ )
         {
            int inputrowoffset = ex*filters*srcHeight*srcWidth + filt*srcHeight*srcWidth +(y+ty)*srcWidth;
            accum += input[inputrowoffset+(x-2+2)]*filterBank[filt*25+ty*5+0] + 
                     input[inputrowoffset+(x-1+2)]*filterBank[filt*25+ty*5+1] + 
                     input[inputrowoffset+(x-0+2)]*filterBank[filt*25+ty*5+2] + 
                     input[inputrowoffset+(x+1+2)]*filterBank[filt*25+ty*5+3] + 
                     input[inputrowoffset+(x+2+2)]*filterBank[filt*25+ty*5+4];
         }
    
      int dstWidth = srcWidth-5+1;
      int dstHeight = srcHeight-5+1;
      output[ex*dstWidth*dstHeight+y*dstWidth+x] = accum;
   }
}

__global__ void GPUConvolveFilterBankTo2D( float* input, float* filterBank, float* output, mint examples, mint filters, mint height, mint width, mint filterSize )
{
   d_GPUConvolveFilterBankTo2D( input, filterBank, output, examples, filters, height, width, filterSize );
}

__global__ void GPUConvolveFilterBankToFilterBank( float* input, float* filterBanks, float* output, mint examples, mint srcfilters, mint dstfilters, mint srcHeight, mint srcWidth, mint filterSize )
{
   for ( int dfilter = 0 ; dfilter < dstfilters ; dfilter++ )
      d_GPUConvolveFilterBankTo2D( input, filterBanks + dfilter*srcfilters*5*5, output + dfilter*examples*(srcHeight-5+1)*(srcWidth-5+1), examples, srcfilters, srcHeight, srcWidth, filterSize );
}

//Note terminology we are writing into src using info from dst
__global__ void GPUBackPropogateConvolveFilterBankToFilterBank( float* input, float* filterBanks, float* output, mint examples, mint srcfilters, mint dstfilters, mint srcHeight, mint srcWidth, mint filterSize )
{
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;

   if ( x >= srcWidth ) return;
   if ( y >= srcHeight ) return;

int t = srcfilters*srcHeight*srcWidth;
/*Slightly suspicious as to why I can't put this directly in output
I hope there aren't any memory overspills or race conditions! */

   int dstWidth = srcWidth-5+1;
   int dstHeight = srcHeight-5+1;

   for ( int ex = 0 ; ex < examples ; ex++ )
      for ( int f = 0 ; f < srcfilters ; f++ )
      {
         float accum = 0.0;

   for ( int f2 = 0 ; f2 < dstfilters ; f2++ )

         for ( int ty = 0 ; ty < 5 ; ty++ )
            for ( int tx = 0 ; tx < 5 ; tx++ )
               if ( ( y-ty < 0 ) || ( y-ty >= dstHeight ) || ( x-tx < 0 ) || (x-tx >= dstWidth ) )
                  accum += 0.0;
               else
                  accum += input[ ex*dstfilters*dstWidth*dstHeight + f2*dstWidth*dstHeight + (y-ty)*dstWidth + (x-tx) ] * filterBanks[ f2*srcfilters*5*5 + f*5*5 + ty*5 + tx ];

/* See previous comment on t!!! We get very minor discrepancy in CPU reconciliation */
      output[ ex*t + f*srcHeight*srcWidth + y*srcWidth + x ] = accum;
   }
}

__global__ void GPUBackPropogateMaxConvolveFilterBankToFilterBank( float* input, float* output, float* postLayerDeltaA, float* preLayerDeltaA, mint examples, mint filters, mint height, mint width )
{
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;

   if ( x >= width ) return;
   if ( y >= height ) return;

   for ( int ex = 0 ; ex < examples ; ex++ )
      for ( int f = 0 ; f < filters ; f++ )
      {
         float accum = 0.0;
         for ( int ty = -1 ; ty <= 1 ; ty++ )
            for ( int tx = -1 ; tx <= 1 ; tx++ )
               if ( y+ty >= 0 && y+ty < height && x+tx >=0 && x+tx < width &&
                     input[ ex*filters*height*width + f*height*width + y*width + x ] == output[ex*filters*height*width + f*height*width + (y+ty)*width + x+tx] )
                  accum += postLayerDeltaA[ ex*filters*height*width + f*height*width + (y+ty)*width + x+tx];

         preLayerDeltaA[ ex*filters*height*width + f*height*width + y*width + x] = accum;
      }
}

__global__ void GPUForwardPropogateMaxConvolveFilterBankToFilterBank( float* input, float* output, mint examples, mint filters, mint height, mint width )
{
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;

   if ( x >= width ) return;
   if ( y >= height ) return;

   for ( int ex = 0 ; ex < examples ; ex++ )
      for ( int f = 0 ; f < filters ; f++ )
      {
         float max  = input[ ex*filters*height*width + f*height*width + y*width + x ];

         for ( int ty = -1 ; ty <= 1 ; ty++ )
            for ( int tx = -1 ; tx <= 1 ; tx++ )
               if ( y+ty >= 0 && y+ty < height && x+tx >=0 && x+tx < width &&
                     input[ ex*filters*height*width + f*height*width + (y+ty)*width + x+tx ] > max )
                  max = input[ ex*filters*height*width + f*height*width + (y+ty)*width + x+tx ];

         output[ ex*filters*height*width + f*height*width + y*width + x ] = max;
      }
}

__global__ void GPUConvolve2DToFilterBank( float* input, float* filterBanks, float* output, mint examples, mint filters, mint srcHeight, mint srcWidth, mint filterSize )
{
   for ( int dfilter = 0 ; dfilter < filters ; dfilter++ )
      d_GPUConvolveFilterBankTo2D( input, filterBanks + dfilter*5*5, output + dfilter*examples*(srcHeight-5+1)*(srcWidth-5+1), examples, 1, srcHeight, srcWidth, filterSize );
}

__global__ void GPUMaxPoolingFilterBankToFilterBank( float* input, float* output, mint examples, mint filters, mint srcHeight, mint srcWidth )
{
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;

   if ( x >= srcWidth/2 )
      return;
   if ( y >= srcHeight/2 )
      return;


   for ( int ex = 0 ; ex < examples ; ex++ )
      for ( int filt = 0 ; filt < filters ; filt++ )
      {
         float v1 = input[ ex*32*srcHeight*srcWidth + filt*srcHeight*srcWidth +(y*2+0)*srcWidth + 2*x + 0 ];
         float v2 = input[ ex*32*srcHeight*srcWidth + filt*srcHeight*srcWidth +(y*2+0)*srcWidth + 2*x + 1 ];
         float v3 = input[ ex*32*srcHeight*srcWidth + filt*srcHeight*srcWidth +(y*2+1)*srcWidth + 2*x + 0 ];
         float v4 = input[ ex*32*srcHeight*srcWidth + filt*srcHeight*srcWidth +(y*2+1)*srcWidth + 2*x + 1 ];

         float max = v1;
         if ( v2 > max ) max = v2;
         if ( v3 > max ) max = v3;
         if ( v4 > max ) max = v4;

    
         int dstWidth = srcWidth/2;
         int dstHeight = srcHeight/2;
         output[ex*32*dstWidth*dstHeight + filt*dstWidth*dstHeight + y*dstWidth + x] = max;
      }
}
";


CNGPUInitialize[]:=
( 
   GPUConvolveFilterBankTo2DFn = CUDAFunctionLoad[GPUCode , "GPUConvolveFilterBankTo2D", {{"Float","Input"},{"Float","Input"},{"Float",_,"Output"},_Integer,_Integer,_Integer,_Integer,_Integer},{28,28}];
   GPUConvolveFilterBankToFilterBankFn = CUDAFunctionLoad[GPUCode , "GPUConvolveFilterBankToFilterBank", {{"Float","Input"},{"Float","Input"},{"Float",_,"Output"},_Integer,_Integer,_Integer,_Integer,_Integer,_Integer},{28,28}];
   GPUBackPropogateConvolveFilterBankToFilterBankFn = CUDAFunctionLoad[GPUCode , "GPUBackPropogateConvolveFilterBankToFilterBank", {{"Float","Input"},{"Float","Input"},{"Float",_,"Output"},_Integer,_Integer,_Integer,_Integer,_Integer,_Integer},{28,28}];
   GPUConvolve2DToFilterBankFn = CUDAFunctionLoad[GPUCode , "GPUConvolve2DToFilterBank", {{"Float","Input"},{"Float","Input"},{"Float",_,"Output"},_Integer,_Integer,_Integer,_Integer,_Integer},{28,28}];
   GPUMaxPoolingFilterBankToFilterBankFn = CUDAFunctionLoad[GPUCode , "GPUMaxPoolingFilterBankToFilterBank", {{"Float","Input"},{"Float",_,"Output"},_Integer,_Integer,_Integer,_Integer},{28,28}];
   GPUBackPropogateMaxConvolveFilterBankToFilterBankFn = CUDAFunctionLoad[GPUCode , "GPUBackPropogateMaxConvolveFilterBankToFilterBank", {{"Float","Input"},{"Float","Input"},{"Float","Input"},{"Float",_,"Output"},_Integer,_Integer,_Integer,_Integer},{28,28}];
   GPUForwardPropogateMaxConvolveFilterBankToFilterBankFn = CUDAFunctionLoad[GPUCode , "GPUForwardPropogateMaxConvolveFilterBankToFilterBank", {{"Float","Input"},{"Float",_,"Output"},_Integer,_Integer,_Integer,_Integer},{28,28}];
)


CNForwardPropogateLayer[GPUConvolveFilterBankTo2D,inputs_] := ( 
   input=CUDAMemoryLoad[Flatten[inputs],"Float"];
   output=CUDAMemoryAllocate["Float",Length[inputs]*(Length[inputs[[1,1]]]-5+1)  * (Length[inputs[[1,1,1]]]-5+1)];
   filterBank = Table[Random[],{32},{5},{5}];
   gpuFilterBank = CUDAMemoryLoad[Flatten[filterBank],"Float"];
   GPUConvolveFilterBankTo2DFn[ input, gpuFilterBank, output, Length[inputs], Length[inputs[[1]]], Length[inputs[[1,1]]], Length[inputs[[1,1,1]]] , 5, 28*28];
   res=unflatten[CUDAMemoryGet[output],{100,28,28}];
   CUDAMemoryUnload[output];CUDAMemoryUnload[input];CUDAMemoryUnload[gpuFilterBank];
   res
);
CNBackPropogateLayer[GPUConvolveFilterBankTo2D[bias_,kernels_],postLayerDeltaA_,_,_] := CNBackPropogateLayer[ConvolveFilterBankTo2D[bias,kernels],postLayerDeltaA,_,_];
CNGradLayer[GPUConvolveFilterBankTo2D[bias_,kernels_],layerInputs_,layerOutputDelta_] := CNGradLayer[ConvolveFilterBankTo2D[bias,kernels],layerInputs,layerOutputDelta];
CNLayerWeightPlus[GPUConvolveFilterBankTo2D[bias_,kernels_],grad_] := ReplacePart[CNLayerWeightPlus[ConvolveFilterBankTo2D[bias,kernels],grad],0->GPUConvolveFilterBankTo2D]


CNForwardPropogateLayer[GPUConvolveFilterBankToFilterBank[filters_],inputs_] := ( 
   input=CUDAMemoryLoad[Flatten[inputs],"Float"];
   srcWidth = Length[inputs[[1,1,1]]];
   srcHeight = Length[inputs[[1,1]]];
   outputWidth = Length[inputs[[1,1,1]]]-5+1;
   outputHeight = Length[inputs[[1,1]]]-5+1;
   dstFilters = Length[filters];
   output=CUDAMemoryAllocate["Float", Length[inputs] * dstFilters * outputHeight  * outputWidth];
   gpuFilterBanks = CUDAMemoryLoad[st=Flatten[filters[[All,2]]],"Float"];
   GPUConvolveFilterBankToFilterBankFn[ input, gpuFilterBanks, output, Length[inputs], Length[inputs[[1]]], dstFilters, srcHeight, srcWidth , 5, {outputWidth,outputHeight}];

   res=unflatten[CUDAMemoryGet[output],{dstFilters,Length[inputs],outputHeight,outputWidth}];
   CUDAMemoryUnload[output];CUDAMemoryUnload[input];CUDAMemoryUnload[gpuFilterBanks];
   Transpose[MapThread[#1+#2&,{res,filters[[All,1]]}],{2,1,3,4}]
);
CNBackPropogateLayer[ GPUConvolveFilterBankToFilterBank[filters_], postLayerDeltaA_, _, _] := (*CNBackPropogateLayer[ConvolveFilterBankToFilterBank[filters], postLayerDeltaA, inputs, outputs];*) (
   input=CUDAMemoryLoad[Flatten[postLayerDeltaA],"Float"];
(* Note terminology src and dst refer to inputs and outputs of the layer respectively, so our algo will be backpropogating from dst to src, ie writes occur in src *)
   srcWidth = Length[postLayerDeltaA[[1,1,1]]]+5-1;
   srcHeight = Length[postLayerDeltaA[[1,1]]]+5-1;
   dstWidth = Length[postLayerDeltaA[[1,1,1]]];
   dstHeight = Length[postLayerDeltaA[[1,1]]];
   srcFilters = Length[filters[[1,2]]];
   dstFilters = Length[filters];
   output=CUDAMemoryAllocate["Float", Length[postLayerDeltaA] * srcFilters * srcHeight  * srcWidth];
   gpuFilterBanks = CUDAMemoryLoad[st=Flatten[filters[[All,2]]],"Float"];
   GPUBackPropogateConvolveFilterBankToFilterBankFn[ input, gpuFilterBanks, output, Length[postLayerDeltaA], srcFilters, dstFilters, srcHeight, srcWidth , 5, {srcWidth,srcHeight}];
   res=unflatten[CUDAMemoryGet[output],{Length[postLayerDeltaA],srcFilters,srcHeight,srcWidth}];
   CUDAMemoryUnload[output];CUDAMemoryUnload[input];CUDAMemoryUnload[gpuFilterBanks];
   res
);
CNGradLayer[ GPUConvolveFilterBankToFilterBank[filters_], layerInputs_, layerOutputDelta_] := CNGradLayer[ConvolveFilterBankToFilterBank[filters],layerInputs,layerOutputDelta];
CNLayerWeightPlus[ GPUConvolveFilterBankToFilterBank[filters_], grad_] := ReplacePart[CNLayerWeightPlus[ConvolveFilterBankToFilterBank[filters],grad], 0->GPUConvolveFilterBankToFilterBank];


CNForwardPropogateLayer[GPUConvolve2DToFilterBank[filters_],inputs_] := ( 
   input=CUDAMemoryLoad[Flatten[inputs],"Float"];
   srcWidth = Length[inputs[[1,1]]];
   srcHeight = Length[inputs[[1]]];
   outputWidth = Length[inputs[[1,1]]]-5+1;
   outputHeight = Length[inputs[[1]]]-5+1;
   dstFilters = Length[filters];
   output=CUDAMemoryAllocate["Float", Length[inputs] * dstFilters * outputHeight  * outputWidth];
   gpuFilterBanks = CUDAMemoryLoad[st=Flatten[de=filters[[All,2]]],"Float"];
   GPUConvolve2DToFilterBankFn[ input, gpuFilterBanks, output, Length[inputs], dstFilters, srcHeight, srcWidth , 5, {outputWidth,outputHeight}];

   res=unflatten[CUDAMemoryGet[output],{dstFilters,Length[inputs],outputHeight,outputWidth}];
   CUDAMemoryUnload[output];CUDAMemoryUnload[input];CUDAMemoryUnload[gpuFilterBanks];
   Transpose[MapThread[#1+#2&,{res,filters[[All,1]]}],{2,1,3,4}]
);
CNBackPropogateLayer[ GPUConvolve2DToFilterBank[filters_],postLayerDeltaA_,inputs_,outputs_] := 
   CNBackPropogateLayer[Convolve2DToFilterBank[filters],postLayerDeltaA,inputs,outputs];
CNGradLayer[ GPUConvolve2DToFilterBank[filters_],layerInputs_,layerOutputDelta_] :=
   CNGradLayer[Convolve2DToFilterBank[filters],layerInputs,layerOutputDelta];
CNLayerWeightPlus[GPUConvolve2DToFilterBank[filters_],grad_] :=
   ReplacePart[CNLayerWeightPlus[Convolve2DToFilterBank[filters],grad],0->GPUConvolve2DToFilterBank];


CNForwardPropogateLayer[GPUMaxPoolingFilterBankToFilterBank, inputs_] := (
   srcHeight = Length[inputs[[1,1]]];
   srcWidth = Length[inputs[[1,1,1]]];
   outputWidth = Floor[Length[inputs[[1,1,1]]]/2];
   outputHeight = Floor[Length[inputs[[1,1]]]/2];
   input = CUDAMemoryLoad[Flatten[inputs],"Float"];
   output = CUDAMemoryAllocate[ "Float", Length[inputs] * Length[inputs[[1]]] * outputHeight * outputWidth ];
   GPUMaxPoolingFilterBankToFilterBankFn[ input, output, Length[inputs], Length[inputs[[1]]], srcHeight, srcWidth, {outputWidth,outputHeight} ];
   res = unflatten[CUDAMemoryGet[output], {Length[inputs], Length[inputs[[1]]],outputHeight,outputWidth}];
   CUDAMemoryUnload[output];CUDAMemoryUnload[input];
   res);
CNBackPropogateLayer[ GPUMaxPoolingFilterBankToFilterBank,postLayerDeltaA_,layerInputs_, layerOutputs_] :=
   CNBackPropogateLayer[MaxPoolingFilterBankToFilterBank,postLayerDeltaA,layerInputs,layerOutputs];
CNGradLayer[ GPUMaxPoolingFilterBankToFilterBank,layerInputs_,layerOutputDelta_] :=
   CNGradLayer[MaxPoolingFilterBankToFilterBank,layerInputs,layerOutputDelta];
CNLayerWeightPlus[ GPUMaxPoolingFilterBankToFilterBank, grad_] := ReplacePart[CNLayerWeightPlus[MaxPoolingFilterBankToFilterBank,grad],0->GPUMaxPoolingFilterBankToFilterBank];


CNForwardPropogateLayer[GPUMaxConvolveFilterBankToFilterBank, inputs_] := (
   gpuinputs = CUDAMemoryLoad[Flatten[inputs],"Float"];
   gpuOutput = CUDAMemoryAllocate[ "Float", Length[inputs] * Length[inputs[[1]]] * Length[inputs[[1,1]]] * Length[inputs[[1,1,1]]] ];
   GPUForwardPropogateMaxConvolveFilterBankToFilterBankFn[ gpuinputs, gpuOutput, Length[inputs], Length[inputs[[1]]], Length[inputs[[1,1]]], Length[inputs[[1,1,1]]], { Length[inputs[[1,1]]],  Length[inputs[[1,1,1]]] } ];
   res = unflatten[CUDAMemoryGet[gpuOutput], {Length[inputs], Length[inputs[[1]]],Length[inputs[[1,1]]],Length[inputs[[1,1,1]]]}];
   CUDAMemoryUnload[gpuinputs];CUDAMemoryUnload[gpuOutput];
   res); 
CNBackPropogateLayer[ GPUMaxConvolveFilterBankToFilterBank,postLayerDeltaA_,layerInputs_, layerOutputs_] := (
   gpuLayerInputs = CUDAMemoryLoad[Flatten[layerInputs],"Float"];
   gpuLayerOutputs = CUDAMemoryLoad[Flatten[layerOutputs],"Float"];
   gpuPostLayerDeltaA = CUDAMemoryLoad[Flatten[postLayerDeltaA],"Float"];
   gpuOutput = CUDAMemoryAllocate[ "Float", Length[layerInputs] * Length[layerInputs[[1]]] * Length[layerInputs[[1,1]]] * Length[layerInputs[[1,1,1]]] ];
   GPUBackPropogateMaxConvolveFilterBankToFilterBankFn[ gpuLayerInputs, gpuLayerOutputs, gpuPostLayerDeltaA, gpuOutput, Length[layerInputs], Length[layerInputs[[1]]], Length[layerInputs[[1,1]]], Length[layerInputs[[1,1,1]]], { Length[layerInputs[[1,1]]],  Length[layerInputs[[1,1,1]]] } ];
   res = unflatten[CUDAMemoryGet[gpuOutput], {Length[layerInputs], Length[layerInputs[[1]]],Length[layerInputs[[1,1]]],Length[layerInputs[[1,1,1]]]}];
   CUDAMemoryUnload[gpuLayerInputs];CUDAMemoryUnload[gpuLayerOutputs];CUDAMemoryUnload[gpuPostLayerDeltaA];CUDAMemoryUnload[gpuOutput];
   res); 
CNGradLayer[ GPUMaxConvolveFilterBankToFilterBank,layerInputs_,layerOutputDelta_] :=
   CNGradLayer[MaxConvolveFilterBankToFilterBank,layerInputs,layerOutputDelta];
CNLayerWeightPlus[ GPUMaxConvolveFilterBankToFilterBank, grad_] := ReplacePart[CNLayerWeightPlus[MaxConvolveFilterBankToFilterBank,grad],0->GPUMaxConvolveFilterBankToFilterBank];


CNConvertCPUToGPU[net_] := Map[CNConvertCPUToGPULayer,net];
CNConvertCPUToGPULayer[ConvolveFilterBankToFilterBank[ filters_ ] ] := GPUConvolveFilterBankToFilterBank[ filters ];
CNConvertCPUToGPULayer[Convolve2DToFilterBank[ filters_ ] ] := GPUConvolve2DToFilterBank[ filters ];
CNConvertCPUToGPULayer[MaxPoolingFilterBankToFilterBank ] := GPUMaxPoolingFilterBankToFilterBank;
CNConvertCPUToGPULayer[MaxConvolveFilterBankToFilterBank ] := GPUMaxConvolveFilterBankToFilterBank;
CNConvertCPUToGPULayer[layer_] := layer;
