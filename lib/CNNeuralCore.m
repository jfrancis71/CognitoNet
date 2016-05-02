(* ::Package:: *)

(*
   Main file where all the central neural network logic is contained.
   The actual neural network layers are defined in file CNNeuralLayers.m
*)


<<CNUtils.m
<<CNNeuralLayers.m


CNReadModel::usage = "CNReadModel[\"file\"] reads in a pretrained file.\n
It uses the $CNModelDir variable to determine the base directory to search from.
The .wdx extension is assumed and should not be appended.\n
It will write over the TrainingHistory,ValidationHistory,CurrentModel and LearningRate variables.
";
CNReadModel[netFile_String]:=(
   {TrainingHistory,ValidationHistory,CurrentModel,CurrentLearningRate} = 
      Import[$CNModelDir<>netFile<>".wdx"];);


CNImageQ[image_] := ImageQ[image]&&ImageChannels[image]==1;


CNColImageQ[image_] := ImageQ[image]&&ImageChannels[image]==3;


CNImageListQ[images_List] := ImageQ[images[[1]]]&&ImageChannels[images[[1]]]==1


CNColImageListQ[images_List] := ImageQ[images[[1]]]&&ImageChannels[images[[1]]]==3


CNDescription::usage = "CNDescription[network] returns a description of the network.";
CNDescription[network_] := Column[{
   Map[
      CNLayerDescription[#]<>" with " <> ToString[CNLayerNumberParameters[#]] <> " parameter(s)"&,network]//MatrixForm,
   "Total of " <> ToString[Total[Map[CNLayerNumberParameters,network]]] <> " parameters."
   }];
CNDescription[network_,input_] := Module[
   {forward = Rest[CNForwardPropogateLayers[{input},network]]},df=forward;
   Column[{
   MapThread[
      CNLayerDescription[#1]<>" with " <> 
         ToString[CNLayerNumberParameters[#1]] <> " parameter(s) and " <> 
         ToString[Length[Flatten[#2]]] <> " neurons"&,
      {network,forward}]//MatrixForm,
   "Total of " <> ToString[Total[Map[CNLayerNumberParameters,network]]] <> 
      " parameters and " <> ToString[Length[Flatten[forward]]] <> 
      " neurons."
   }]];


(* Forward Propogation Logic *)


(* We partition at this point, as on some large datasets and networks we will use
   an excessive amount of memory otherwise. Functionally it is the same as calling
   ForwardPropogateInternal.
*)
CNForwardPropogate::usage=
   "CNForwardPropogate[inputs,network] runs forward propogation on the inputs through the network.
CNForwardPropogate[image,network] runs forward propogation on the image through the network.
CNForwardPropogate[images,network] runs forward propogation on the images through the network.";
CNForwardPropogate[inputs_,network_]:=
   Flatten[Map[CNForwardPropogateInternal[#,network]&,Partition[inputs,100,100,1,{}]],1];


CNForwardPropogate[images_?CNImageListQ,network_] :=
   CNForwardPropogate[Map[ImageData[#]&,images],network];


CNForwardPropogate[images_?CNColImageListQ,network_] :=
   CNForwardPropogate[Map[ImageData[#,Interleaving->False]&,images],network];


CNForwardPropogate[image_?CNImageQ,network_] :=
   CNForwardPropogate[ {ImageData[image]}, network ][[1]];


CNForwardPropogate[image_?CNColImageQ,network_] :=
   CNForwardPropogate[ {ImageData[image,Interleaving->False]}, network ][[1]];


(* Note you should be cautious using this function applied to large datasets with
   a large and complex neural network. Memory usage may be excessive.
   Consider using the partition'd version ForwardPropogate instead
*)
CNForwardPropogateInternal[inputs_List,network_]:=
   Last[CNForwardPropogateLayers[inputs,network]];


CNForwardPropogateLayers::usage = "CNForwardPropogateLayers[inputs,network] runs inputs through the network
and returns the output of each layer. Note you should be cautious using this with large multi layer nets with a large
number of input examples due to excessive memory usage. (Consider using smaller #inputs)";
CNForwardPropogateLayers[inputs_List,network_]:=
   Rest[FoldList[CNForwardPropogateLayer[#2,#1]&,inputs,network]];


(* Classification Logic *)


CNClassifyToIndex::usage = "CNClassifyToIndex[inputs,network] will run the inputs through the network
and output the index of the most likely category for each example.
It assumes the network outputs probabilities in a 1 of K format.";
CNClassifyToIndex[inputs_List,network_List]:=
   Module[
      {outputs=CNForwardPropogate[inputs,network]},
      Table[Position[outputs[[t]],Max[outputs[[t]]]][[1,1]],{t,1,Length[inputs]}]];
CNClassifyToIndex[input_,network_List]:=
   Module[
      {outputs=CNForwardPropogate[input,network]},
      Position[outputs,Max[outputs]][[1,1]]];


CNClassify::usage = "CNClassify[inputs,network,categoryLabels]
Runs the inputs through the neural networks and classifies them according to the most likely category.
The network is expected to output probabilities in a 1 of K format.
Therefore categoryLabels should be a list of length K.";
CNClassify[inputs_List,network_List,categoryLabels_]:=
   Map[categoryLabels[[#]]&,CNClassifyToIndex[inputs,network]];
CNClassify[inputs_,network_List,categoryLabels_]:=
   categoryLabels[[CNClassifyToIndex[inputs,network]]];


CNClassificationPerformance::usage = "
CNClassificationPerformance[inputs,targetLabels,net,categoryMap]
returns fraction of input examples labelled correctly.
CNClassificationPerformance[testSet,net,categoryMap]
returns fraction of test set labelled correctly.
";
CNClassificationPerformance[inputs_,targetLabels_,net_,categoryMap_List]:=
   Total[Boole[MapThread[Equal,{CNClassify[inputs,net,categoryMap],targetLabels}]]]/Length[inputs]//N;
CNClassificationPerformance[testSet_,net_,categoryMap_List]:=
   Total[Boole[MapThread[Equal,{CNClassify[testSet[[All,1]],net,categoryMap],testSet[[All,2]]}]]]/Length[testSet]//N;


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


CNRegressionLoss[model_,testSet_] :=
   (outputs=CNForwardPropogate[testSet[[All,1]],model];CNAssertAbort[Dimensions[outputs]==Dimensions[testSet[[All,2]]],"Loss1D::Mismatched Targets and Outputs"];Total[(outputs-testSet[[All,2]])^2,2]/Length[testSet]);
CNRegressionLoss1D[model_,testSet_] :=
   (outputs=CNForwardPropogate[testSet[[All,1]],model];CNAssertAbort[Dimensions[outputs]==Dimensions[testSet[[All,2]]],"Loss1D::Mismatched Targets and Outputs"];Total[(outputs-testSet[[All,2]])^2,2]/Length[testSet]);
CNCrossEntropyLoss[model_,testSet_]:=
   Module[{output=CNForwardPropogate[testSet[[All,1]],model]},re=output;-Total[testSet[[All,2]]*Log[output]+(1-testSet[[All,2]])*Log[1-output],2]/Length[testSet]];
CNClassificationLoss[parameters_,testSet_]:=-Total[Log[Extract[CNForwardPropogate[testSet[[All,1]],parameters],Position[testSet[[All,2]],1]]]]/Length[testSet];
