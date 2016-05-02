(* ::Package:: *)

(* Training Logic *)


CNLayerWeightPlus[ networkLayers_List, grad_List ] :=MapThread[ CNLayerWeightPlus, { networkLayers, grad } ]


CNBackPropogateLayers::usage = "CNBackPropogateLayers[ network, neuronActivations, finalLayerDelta ] will backpropogate the
sensitivity of the loss function given by finalLayerDelta through the network. neuronActivations is the current activations
for all the neurons in the network and for all the training examples in the gradient descent calculation. So the number of
examples in finalLayerDelta should match the number of examples in neuronActivations.
It stops before backpropogating to the input layer as this is rarely needed.";
(*
   The algorith architecture is to bakcpropogate starting at the final layer and going backwards.
   The delta's ( or loss error ) for each layer are computed with the function CNBackPropogateLayer. It needs to know the
   delta's for that layer, and the neuron activations for that layer and also the neuron activations for the previous layer.
   So note, when you call CNBackPropogateLayer on a layer, it does not compute the delta's for that layer. It is using it's
   knowledge of that layer, the delta's for that layer and the activations at that layer (and the previous layer) to compute
   what the delta's are for the previous layer.
*)
CNBackPropogateLayers[model_,neuronActivations_,finalLayerDelta_,OptionsPattern[]]:=(
  
   networkLayers=Length[model];

   delta = Table[$Failed,{Length[model]}];
   delta[[-1]] = finalLayerDelta;

   For[layerIndex=networkLayers,layerIndex>1,layerIndex--,
(* layerIndex refers to layer being back propogated across
   ie computing delta's for layerIndex-1 given layerIndex *)

      CNTimer["Backprop Layer "<>CNLayerDescription[model[[layerIndex]]],
         delta[[layerIndex-1]]=
           CNBackPropogateLayer[
               model[[layerIndex]],
               delta[[layerIndex]],
               neuronActivations[[layerIndex-1]],
               neuronActivations[[layerIndex]]];
      ];

      CNAssertAbort[Dimensions[delta[[layerIndex-1]]]==Dimensions[neuronActivations[[layerIndex-1]]]];
      (*delta[[layerIndex-1]]+=Sign[neuronActivations[[layerIndex-1]]]*xL1A;*)
   ];

   delta
)


CNGrad::usage = "CNGrad[ network, inputs, targets, lossF ] computes the gradient of the parameters
in a neural network";
CNGrad[model_,inputs_,targets_,lossF_]:=(

   CNAssertAbort[Length[inputs]==Length[targets],
      "CNGrad::# of Training Labels should equal # of Training Inputs"];

   L = CNTimer["CNForwardPropogateLayers",CNForwardPropogateLayers[inputs, model]];
   CNAssertAbort[Dimensions[L[[-1]]]==Dimensions[targets],
      "NNGrad::Dimensions of outputs and targets should match"];

   CNTimer["BackPropogate Total",
      xDelta = CNBackPropogateLayers[
         model,
         L,
         CNDeltaLoss[lossF,L[[-1]],targets]];];

(* We seperate out the final stage as there's no L[[0]], we get that from the inputs *)
   CNTimer["LayerGrad",
   Prepend[
      Table[
         CNTimer["LayerGrad::"<>CNLayerDescription[model[[layerIndex]]],CNGradLayer[model[[layerIndex]],L[[layerIndex-1]],xDelta[[layerIndex]]]]
         ,{layerIndex,2,Length[model]}],
      CNGradLayer[model[[1]],inputs,xDelta[[1]]]
   ]]
);


CNTrainModel::usage = "CNTrainModel[ network, trainingSet, lossF, opts ] trains a
neural network by gradient descent (not mini batch).
Options are:
   MaxEpoch->1000
   LearningRate->.01
   MomentumDecay->0
   MomentumType->None
";
SyntaxInformation[MaxEpoch]={"ArgumentsPattern"->{_}};
SyntaxInformation[LearningRate]={"ArgumentsPattern"->{_}};
SyntaxInformation[EpochMonitor]={"ArgumentsPattern"->{_}};
Options[CNTrainModel]={
   MaxEpoch->1000,
   LearningRate->.01,
   MomentumDecay->.0,
   MomentumType->"None",
   EpochMonitor:>(#&)};
CNTrainModel[model_,trainingSet_,lossF_,opts:OptionsPattern[]] := Module[{grOutput={}},
   TrainingHistory = {};ValidationHistory={};CurrentLearningRate=OptionValue[LearningRate];
   Print[Dynamic[grOutput]];
   CNGradientDescent[
      model,
      CNGrad[#,trainingSet[[All,1]],trainingSet[[All,2]],lossF]&,
      CNLayerWeightPlus,
      OptionValue[MaxEpoch],
      {
         StepSize->OptionValue[LearningRate],
         MomentumDecay->OptionValue[MomentumDecay],
         MomentumType->OptionValue[MomentumType],
         StepMonitor:>Function[currentState,CurrentModel=currentState;AppendTo[TrainingHistory,lossF[currentState,trainingSet]];grOutput=ListPlot[TrainingHistory];OptionValue[EpochMonitor][currentState]]}
       ]
   ]


Options[CNMiniBatchTrainForOneEpoch]={
   LearningRate->.01,
   MomentumDecay->.0,
   MomentumType->"None"
};
CNMiniBatchTrainForOneEpoch[ {network_, velocity_ }, trainingSet_, lossF_, opts:OptionsPattern[] ] := (
   { state, vel } = { network, velocity };
   Scan[
      Function[batch,{state,vel}=CNStepGradientDescent[{state, vel},CNGrad[ #, batch[[All,1]],batch[[All,2]], lossF ]&, CNLayerWeightPlus,OptionValue[MomentumDecay],OptionValue[MomentumType],OptionValue[LearningRate]];partialTrainingLoss = lossF[ state, batch];],
      Partition[trainingSet,100,100,1,{}]
   ];
   { state, vel }
);


CNMiniBatchTrainModel::usage = "CNMiniBatchTrainModel[ network, trainingSet, lossF, opts ] trains a
neural network by gradient descent (not mini batch).
Options are:
   MaxEpoch->1000
   LearningRate->.01
   MomentumDecay->0
   MomentumType->None
";
Options[CNMiniBatchTrainModelInternal]={
   MaxEpoch->10,
   LearningRate->.01,
   MomentumDecay->.0,
   MomentumType->"None",
   ValidationSet->{},
   EpochMonitor:>(#&)};
CNMiniBatchTrainModelInternal[model_,trainingSet_,lossF_,opts:OptionsPattern[]] := Module[{grOutput={}},
   TrainingHistory = {};ValidationHistory={};CurrentLearningRate=OptionValue[LearningRate];
   grOutput=0;
   partialTrainingLoss = Null; Print[" Batch Training Loss: ",Dynamic[partialTrainingLoss ]];
   Print[Dynamic[grOutput]];
   (* Last term below needed to ensure velocity has the right structure *)
   {state,velocity} = { model, CNGrad[ model, trainingSet[[1;;1,1]], trainingSet[[1;;1,2]], CNRegressionLoss1D ]*0.0 };
   For[epoch=1,epoch<OptionValue[MaxEpoch],epoch++,
      { state, velocity } = CNMiniBatchTrainForOneEpoch[ {state, velocity}, trainingSet, lossF, FilterRules[{opts},Options[CNMiniBatchTrainForOneEpoch]] ];
      CurrentModel = state;
      AppendTo[TrainingHistory,lossF[ state, trainingSet ] ];
      If[Length[OptionValue[ValidationSet]]>0,
         AppendTo[ValidationHistory, lossF[ state, OptionValue[ ValidationSet ] ] ] ];
      grOutput = If[
         Length[OptionValue[ValidationSet]]>0,
            ListPlot[{TrainingHistory,ValidationHistory}],
            ListPlot[TrainingHistory]];
   ];
];

Options[CNMiniBatchTrainModel]={
   MaxEpoch->10,
   LearningRate->.01,
   MomentumDecay->.0,
   MomentumType->"None",
   ValidationSet->{},
   EpochMonitor:>(#&)};
CNMiniBatchTrainModel[model_,trainingSet_,lossF_,opts:OptionsPattern[]] :=
   Which[
      CNImageQ[trainingSet[[1,1]]], CNMiniBatchTrainModelInternal[ model, Map[ImageData[#[[1]]]->#[[2]]&, trainingSet], lossF, opts ],
      CNColImageQ[trainingSet[[1,1]]], CNMiniBatchTrainModelInternal[ model, Map[ImageData[#[[1]],Interleaving->False]->#[[2]]&,trainingSet], lossF, opts ],
      True, CNMiniBatchTrainModelInternal[ model, trainingSet, lossF, opts ]
   ];


CNConvertTargetsTo1OfKRepresentation[targets_,categoryList_]:=
   Map[
      ReplacePart[ConstantArray[0,Length[categoryList]],Position[categoryList,#][[1,1]]->1]&,targets];


CNConvertTrainingSetTo1OfKRepresentation[trainingSet_,categoryList_]:=
   Map[#[[1]]->
      ReplacePart[ConstantArray[0,Length[categoryList]],Position[categoryList,#[[2]]][[1,1]]->1]&,trainingSet];


Options[CNMiniBatchTrainCategoricalModel]={
   MaxEpoch->10,
   LearningRate->.01,
   MomentumDecay->.0,
   MomentumType->"None",
   ValidationSet->{},
   EpochMonitor:>(#&)};
CNMiniBatchTrainCategoricalModel[model_,trainingSet_,lossF_, categoryList_, opts:OptionsPattern[]] :=
   CNMiniBatchTrainModel[ model, CNConvertTrainingSetTo1OfKRepresentation[trainingSet, categoryList], lossF, opts ]


CNCheckGrad[weight_,network_, testSet_,lossF_,epsilon_:10^-6]:=
   (lossF[ReplacePart[network,weight->Extract[network,weight]+epsilon], testSet]-
   lossF[network, testSet])/epsilon


(* Loss Functions *)


CNDeltaLoss[CNRegressionLoss,outputs_,targets_]:=2.0*(outputs-targets)/Length[outputs];
CNDeltaLoss[CNRegressionLoss1D,outputs_,targets_]:=2.0*(outputs-targets)/Length[outputs];
CNDeltaLoss[CNCrossEntropyLoss,outputs_,targets_]:=-((-(1-targets)/(1-outputs)) + (targets/outputs))/Length[outputs];
CNDeltaLoss[CNClassificationLoss,outputs_,targets_]:=-targets*(1.0/outputs)/Length[outputs];
