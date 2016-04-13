(* ::Package:: *)

(*
   Main file where all the central neural network logic is contained.
   The actual neural network layers are defined in file CNNeuralLayers.m
*)


<<CNUtils.m
<<CNNeuralLayers.m


CNRead::usage = "CNRead[\"file\"] reads in a pretrained file.\n
It uses the $CNModelDir variable to determine the base directory to search from.
The .wdx extension is assumed and should not be appended.\n
It will write over the TrainingHistory,ValidationHistory,CurrentModel and LearningRate variables.
";
CNRead[netFile_String]:=(
   {TrainingHistory,ValidationHistory,CurrentModel,LearningRate} = 
      Import[$CNModelDir<>netFile<>".wdx"];);


(* We partition at this point, as on some large datasets and networks we will use
   an excessive amount of memory otherwise. Functionally it is the same as calling
   ForwardPropogateInternal.
*)
CNForwardPropogate::usage=
   "CNForwardPropogate[inputs,network] runs forward propogation on the inputs through the network.";
CNForwardPropogate[inputs_List,network_]:=
   Flatten[Map[CNForwardPropogateInternal[#,network]&,Partition[inputs,100,100,1,{}]],1];


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
   FoldList[CNForwardPropogateLayer[#2,#1]&,inputs,network];


CNClassifyToIndex::usage = "CNClassifyToIndex[inputs,network] will run the inputs through the network
and output the index of the most likely category for each example.
It assumes the network outputs probabilities in a 1 of K format.";
CNClassifyToIndex[inputs_List,network_List]:=
   Module[
      {outputs=CNForwardPropogate[inputs,network]},
      Table[Position[outputs[[t]],Max[outputs[[t]]]][[1,1]],{t,1,Length[inputs]}]];


CNClassify::usage = "CNClassify[inputs,network,categoryLabels]
Runs the inputs through the neural networks and classifies them according to the most likely category.
The network is expected to output probabilities in a 1 of K format.
Therefore categoryLabels should be a list of length K.";
CNClassify[inputs_List,network_List,categoryLabels_]:=
   Map[categoryLabels[[#]]&,CNClassifyToIndex[inputs,network]];


CNDescription::usage = "CNDescription[network] returns a description of the network.";
CNDescription[network_]:=Map[CNLayerDescription,network];
