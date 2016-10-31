(* ::Package:: *)

<<CNNeuralCore.m


Set::write


trainingSet=Import["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\TrainingSet.mx"];


leftEye=ImageTake[trainingSet[[3,1]],{14-3,14+3},{12-3,12+3}];


rightEye=ImageTake[trainingSet[[3,1]],{14-3,14+3},{24-3,24+3}];


transformedDataSet=Map[{
Image[1-ImageData[ImageCorrelate[#[[1]],leftEye,NormalizedSquaredEuclideanDistance]]],
Image[1-ImageData[ImageCorrelate[#[[1]],rightEye,NormalizedSquaredEuclideanDistance]]
]}->#[[2]]
&,trainingSet[[10;;1010]]];


trF=Map[{
Log[
PDF[NormalDistribution[0.82,.1],ImageData[#[[1,1]]]]/
PDF[NormalDistribution[0.5,.1],ImageData[#[[1,1]]]]],
Log[
PDF[NormalDistribution[0.82,.1],ImageData[#[[1,2]]]]/
PDF[NormalDistribution[0.5,.1],ImageData[#[[1,2]]]]]

}
->#[[2]]&,transformedDataSet];trF[[All,1]]//Dimensions


{1001,2,32,32}


CNForwardPropogateLayer[MyLog,x_]:=Log[x]


CNBackPropogateLayer[MyLog,postLayerDeltaA_,inputs_,_]:=
  postLayerDeltaA/inputs;


CNLayerWeightPlus[MyLog,grad_]:=MyLog;


CNGradLayer[MyLog,layerInputs_,layerOutputDelta_]:={};


CNForwardPropogateLayer[MyAdd[biases_],inputs_]:=Map[#+biases&,inputs];


CNBackPropogateLayer[MyAdd[biases_],postLayerDeltaA_,inputs_,_]:=postLayerDeltaA;


CNLayerWeightPlus[MyAdd[biases_],grad_]:=MyAdd[biases+grad]


CNGradLayer[MyAdd[biases_],layerInputs_,layerOutputDelta_]:=Total[layerOutputDelta];


CNForwardPropogateLayer[MyAddFilterBank[biases_],inputs_]:=Transpose[Table[CNForwardPropogateLayer[MyAdd[biases[[f]]],inputs[[All,f]]],{f,1,Length[inputs[[1]]]}]]


CNBackPropogateLayer[MyAddFilterBank[biases_],postLayerDeltaA_,inputs_,_]:=postLayerDeltaA;


CNLayerWeightPlus[MyAddFilterBank[biases_],grad_]:=MyAddFilterBank[biases+grad]


CNGradLayer[MyAddFilterBank[biases_],layerInputs_,layerOutputDelta_]:=Table[CNGradLayer[MyAdd[biases[[f]]],layerInputs[[All,f]],layerOutputDelta[[All,f]]],{f,1,Length[layerInputs[[1]]]}];


CNForwardPropogateLayer[MyLogSumExpFilterBank,inputs_]:=Map[CNForwardPropogateLayer[LogSumExp,#]&,inputs];


CNBackPropogateLayer[MyLogSumExpFilterBank,postLayerDeltaA_,inputs_,_]:=Table[CNBackPropogateLayer[LogSumExp,postLayerDeltaA[[All,f]],inputs[[All,f]],_]&,{f,1,Length[inputs[[1]]]}];


CNLayerWeightPlus[MyLogSumExpFilterBank,grad_]:=MyLogSumExpFilterBank


CNGradLayer[MyLogSumExpFilterBank,layerInputs_,layerOutputDelta_]:={};


(*Following code is only good for filters that have no state*)


CNForwardPropogateLayer[FilterBank[filter_],inputs_]:=Transpose[Table[CNForwardPropogateLayer[filter,inputs[[All,f]]],{f,1,Length[inputs[[1]]]}]]


CNBackPropogateLayer[FilterBank[filter_],postLayerDeltaA_,inputs_,outputs_]:=Transpose[Table[CNBackPropogateLayer[filter,postLayerDeltaA[[All,f]],inputs[[All,f]],outputs[[All,f]]],{f,1,Length[inputs[[1]]]}]];


CNLayerWeightPlus[FilterBank[filter_],grad_]:=FilterBank[filter]


CNGradLayer[FilterBank[filter_],layerInputs_,layerOutputDelta_]:={}


CNForwardPropogateLayer[MyPlus1D,inputs_]:=Map[Total,inputs]


CNBackPropogateLayer[MyPlus1D,postLayerDeltaA_,inputs_,outputs_]:=Map[{#,#}&,postLayerDeltaA];


CNLayerWeightPlus[MyPlus1D,grad_]:=MyPlus1D


CNGradLayer[MyPlus1D,layerInputs_,layerOutputDelta_]:={};


CNForwardPropogateLayer[PointwiseLinear[c_,m_],inputs_]:=m*inputs+c;


CNBackPropogateLayer[PointwiseLinear[c_,m_],postLayerDeltaA_,inputs_,outputs_]:=m*inputs*postLayerDeltaA;


CNLayerWeightPlus[PointwiseLinear[c_,m_],grad_]:=PointwiseLinear[c+grad[[1]],m+grad[[2]]]


CNGradLayer[PointwiseLinear[c_,m_],layerInputs_,layerOutputDelta_]:={Total[Flatten[layerOutputDelta]],Total[Flatten[layerInputs*layerOutputDelta]]};


net={
PointwiseLinear[-1,1.2],
MyAddFilterBank[{Table[Log[1/1024],{32},{32}],Table[Log[1/1024],{32},{32}]}],
FilterBank[Adaptor2DTo1D[32]],
FilterBank[LogSumExp],
MyPlus1D,
Logistic
};


EucDistDataSet=Map[{ImageData[#[[1,1]]],ImageData[#[[1,2]]]}->#[[2]]&,transformedDataSet];


CNMiniBatchTrainModel[ net, EucDistDataSet[[1;;900]],CNCrossEntropyLoss,{MaxEpoch->40000,LearningRate->.01,Momentum->0.9,MomentumType->"Nesterov",ValidationSet->EucDistDataSet[[901;;-1]]}]
