(* ::Package:: *)

(*
   Main file where all the central neural network logic is contained.
   The actual neural network layers are defined in file CNNeuralLayers.m
*)


<<CNUtils.m
<<CNNeuralLayers.m


CNRead[netFile_String]:=(
   {TrainingHistory,ValidationHistory,CurrentModel,LearningRate} = 
      Import[$CNModelDir<>netFile<>".wdx"];);


(* We partition at this point, as on some large datasets and networks we will use
   an excessive amount of memory otherwise. Functionally it is the same as calling
   ForwardPropogateInternal.
*)
ForwardPropogate::usage=
   "ForwardPropogate[inputs,network] runs forward propogation on the inputs through the network.";
ForwardPropogate[inputs_List,network_]:=
   Flatten[Map[ForwardPropogateInternal[#,network]&,Partition[inputs,100,100,1,{}]],1];


(* Note you should be cautious using this function applied to large datasets with
   a large and complex neural network. Memory usage may be excessive.
   Consider using the partition'd version ForwardPropogate instead.
*)
ForwardPropogateInternal[inputs_List,network_]:=
   Last[ForwardPropogateLayers[inputs,network]];


ForwardPropogateLayers[inputs_List,network_]:=
   FoldList[ForwardPropogateLayer[#2,#1]&,inputs,network];


CNClassifyToIndex[inputs_List,network_List]:=
   Module[
      {outputs=ForwardPropogate[inputs,network]},
      Table[Position[outputs[[t]],Max[outputs[[t]]]][[1,1]],{t,1,Length[inputs]}]]


CNClassify[inputs_List,network_List,categoryLabels_]:=
   Map[categoryLabels[[#]]&,CNClassifyToIndex[inputs,network]]
