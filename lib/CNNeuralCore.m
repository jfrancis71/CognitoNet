(* ::Package:: *)

(*
   Main file where all the central neural network logic is contained.
   The actual neural network layers are defined in file CNNeuralLayers.m
*)


<<CNUtils.m
<<CNNeuralLayers.m
<<CNNeuralTrain.m


CNToActivations[ input_?CNImageQ ] := ImageData[input];
CNToActivations[ input_?CNColImageQ ] := ImageData[input,Interleaving->False];
CNToActivations[ inputs_?CNImageListQ ] := Map[ImageData,inputs];
CNToActivations[ inputs_?CNColImageListQ ] := Map[ ImageData[#,Interleaving->False]&, inputs ];
CNToActivations[ inputs_ ] := inputs;
CNToActivations[ {} ] := {};


CNReadModel::usage = "CNReadModel[\"file\"] reads in a pretrained file.\n
It uses the $CNModelDir variable to determine the base directory to search from.
The .wdx extension is assumed and should not be appended.\n
";
CNReadModel[netFile_String] := Import[$CNModelDir<>netFile<>".wdx"];


CNDescription::usage = "CNDescription[network] returns a description of the network.";
CNDescription[network_] := Column[{
   Map[
      CNLayerDescription[#]<>" with " <> ToString[CNLayerNumberParameters[#]]
         <> " parameter(s)"&,network]//MatrixForm,
   "Total of " <> ToString[Total[Map[CNLayerNumberParameters,network]]] <> " parameters."
   }];
CNDescription[network_,input_] := Module[
   {forward = CNForwardPropogateLayers[{ CNToActivations[ input ] } ,network]},df=forward;
   Column[{
   MapThread[
      CNLayerDescription[#1]<>" with " <> 
         ToString[CNLayerNumberParameters[#1]] <> " parameter(s) and " <> 
         ToString[Length[Flatten[#2]]] <> " neurons"  <> " of structure " <> ToString[Dimensions[#2[[1]]]]&,
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
   "CNForwardPropogate[inputs,network] runs forward propogation on the inputs through the
network
CNForwardPropogate[image,network] runs forward propogation on the image through the network.
CNForwardPropogate[images,network] runs forward propogation on the images through the network.";
CNForwardPropogate[inputs_,network_]:= (
   CNFDPpg=Partition[inputs,100,100,1,{}]; (* No idea why I can't embed this function directly into
      function below (so no need for local var) but without this Partition will fail non deterministically
      usually within a few thousand calls. Appears to be new behaviour in Mathematica version 11 *)
   Flatten[Map[CNForwardPropogateInternal[#,network]&,CNFDPpg],1])


CNForwardPropogate[images_?(CNImageListQ[#]||CNColImageListQ[#]&),network_] :=
   CNForwardPropogate[ CNToActivations[ images ],network]


CNForwardPropogate[image_?(CNImageQ[#]||CNColImageQ[#]&),network_] :=
   CNForwardPropogate[ { CNToActivations[ image ] }, network ][[1]];


(* Classification Logic *)


CNClassifyToIndex::usage = "CNClassifyToIndex[inputs,network] will run the inputs through the
network and output the index of the most likely category for each example. It assumes the
network outputs probabilities in a 1 of K format.";
CNClassifyToIndex[inputs_List,network_List]:=
   Module[
      {outputs=CNForwardPropogate[inputs,network]},
      Table[Position[outputs[[t]],Max[outputs[[t]]]][[1,1]],{t,1,Length[inputs]}]];
CNClassifyToIndex[input_,network_List]:=
   Module[
      {outputs=CNForwardPropogate[input,network]},
      Position[outputs,Max[outputs]][[1,1]]];


CNClassifyCategoricalModel::usage = "CNClassifyCategoricalModel[inputs,network,categoryLabels]
Runs the inputs through the neural networks and classifies them according to the most likely
category. The network is expected to output probabilities in a 1 of K format. Therefore
categoryLabels should be a list of length K.";
CNClassifyCategoricalModel[inputs_ ,network_List, categoryLabels_]:= (
   CNAssertAbort[ Last[ network ] === Softmax,
      "CNClassifyCategoricalModel::Error - presumably final layer should be Softmax."];
   categoryLabels[[CNClassifyToIndex[inputs,network]]] )


CNCategoricalClassificationAccuracy::usage = "
CNClassificationAccuracy[inputs,targetLabels,net,categoryMap]
returns fraction of input examples labelled correctly.
CNClassificationPerformance[testSet,net,categoryMap]
returns fraction of test set labelled correctly.
";
CNCategoricalClassificationAccuracy[testSet_,net_,categoryMap_List]:=
   Total[Boole[MapThread[Equal,{CNClassifyCategoricalModel[
      testSet[[All,1]],net,categoryMap],testSet[[All,2]]}]]]/Length[testSet]//N;


(*
   ALL Code Below this point has no knowledge of Mathematica Image structures, categories, etc
   It only knows about neural networks, activations and network parameters.
   So any Image objects or category labels must be converted to activations or 1 of K neural
   representations (respectively).
*)


(* Note you should be cautious using this function applied to large datasets with
   a large and complex neural network. Memory usage may be excessive.
   Consider using the partition'd version ForwardPropogate instead
*)
CNForwardPropogateInternal[inputs_List,network_]:=
   Last[CNForwardPropogateLayers[inputs,network]];


CNForwardPropogateLayers::usage = "CNForwardPropogateLayers[inputs,network] runs inputs
through the network and returns the output of each layer. Note you should be cautious using
this with large multi layer nets with a large number of input examples due to excessive
memory usage. (Consider using smaller #inputs)";
CNForwardPropogateLayers[inputs_List,network_]:=
   Rest[FoldList[CNTimer["Forward Layer "<>CNLayerDescription[ #2 ],CNForwardPropogateLayer[#2,#1]]&,inputs,network]];


CNRegressionLoss[model_,testSet_] := (
   outputs=CNForwardPropogate[testSet[[All,1]],model];
   CNAssertAbort[Dimensions[outputs]==Dimensions[testSet[[All,2]]],
      "Loss1D::Mismatched Targets and Outputs"];
   Total[(outputs-testSet[[All,2]])^2,2]/Length[testSet]);
CNRegressionLoss1D[model_,testSet_] := (
   outputs=CNForwardPropogate[testSet[[All,1]],model];
   CNAssertAbort[Dimensions[outputs]==Dimensions[testSet[[All,2]]],
      "Loss1D::Mismatched Targets and Outputs"];
   Total[(outputs-testSet[[All,2]])^2,2]/Length[testSet]);
CNCrossEntropyLossExamples[model_,testSet_] :=
   Module[{output=CNForwardPropogate[testSet[[All,1]],model]},
      -(testSet[[All,2]]*Log[output]+
         (1-testSet[[All,2]])*Log[1-output])];
CNCrossEntropyLoss[model_,testSet_] :=
   Total[ CNCrossEntropyLossExamples[model,testSet] ]/Length[testSet];
CNCategoricalLoss::usage = 
   "CNCategoricalLoss[ parameters, testSet ] Note that no label maps are passed in,
so testSet must be in 1 of K format.";
CNCategoricalLoss[parameters_,testSet_] := -Total[Log[Extract[CNForwardPropogate[
   testSet[[All,1]],parameters],Position[testSet[[All,2]],1]]]]/Length[testSet];
