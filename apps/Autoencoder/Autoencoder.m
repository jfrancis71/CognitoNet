(* ::Package:: *)

(*
References:
1)
Hugo Larochelle Youtube video
Exploring Strategies for Training Deep Neural Networks , Larochelle et al, 2009

2)
Stacking Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion, Vincent, Larochelle et al, 2010

Papers seem to produce different results.
Paper 1 suggests good results learning on MNIST (page 16) they used cross entropy loss
Paper 2 suggests more disappointing results learning on natural images (they used squared loss function). page 3387


*)


(*
Predictive Sparse Decomposition about L1 sparsity
Kavukcuoglu, Ranzato, LeCun 2009
Mentioned in Bay Area Deep Learning School day 1 at 07:29:11
*)





(* ::Input:: *)
(*<<CNNeuralCore.m*)


(* ::Input:: *)
(*{trainingSet,validationSet}=Import["C:\\Users\\julian\\ImageDataSets\\NoEntrySigns5\\DataSetv2.mx"];*)


(* ::Input:: *)
(*dat=Map[ImageData[#][[1;;32,1;;32]]&,Select[trainingSet,#[[2]]==0&][[All,1]]];*)


CNForwardPropogateLayer[Shared[layerBiases_List,layerWeights_?MatrixQ], inputs_List] :=
(
   AutoencoderShared=Shared[layerBiases,layerWeights];
   CNForwardPropogateLayer[xd=FullyConnected1DTo1D[layerBiases,layerWeights],inputs]
 )


(* ::Input:: *)
(*CNBackPropogateLayer[Shared[biases_,weights_],postLayerDeltaA_,_,_] :=*)
(*CNBackPropogateLayer[FullyConnected1DTo1D[biases,weights],postLayerDeltaA,_,_] *)


(* ::Input:: *)
(*CNGradLayer[Shared[biases_,weights_],layerInputs_,layerOutputDelta_] :=*)
(*(fgr=((w1=CNGradLayer[FullyConnected1DTo1D[biases,weights],layerInputs,layerOutputDelta])+(w2={outB[[1;;128]],Transpose[AutoencoderSharedGrad[[2]]]})));*)


CNGradLayer[Shared[biases_,weights_],layerInputs_,layerOutputDelta_] :=
   Module[{fgw=CNGradLayer[FullyConnected1DTo1D[biases,weights],layerInputs,layerOutputDelta]},fgw[[2]] += Transpose[AutoencoderSharedGrad[[2]]];fgw]


(* ::Input:: *)
(*CNLayerWeightPlus[networkLayer_Shared,grad_] :=*)
(*Shared[*)
(*CNLayerWeightPlus[FullyConnected1DTo1D[networkLayer[[1]],networkLayer[[2]]],grad][[1]],*)
(*CNLayerWeightPlus[FullyConnected1DTo1D[networkLayer[[1]],networkLayer[[2]]],grad][[2]]];*)


(* ::Input:: *)
(*CNForwardPropogateLayer[Tied[biases_], inputs_List] :=*)
(* CNForwardPropogateLayer[FullyConnected1DTo1D[biases,Transpose[AutoencoderShared[[2]]]],inputs]*)


(* ::Input:: *)
(*CNBackPropogateLayer[Tied[biases_],postLayerDeltaA_,layerInputs_,_] :=*)
(*(AutoencoderSharedGrad=CNGradLayer[FullyConnected1DTo1D[biases,Transpose[AutoencoderShared[[2]]]],layerInputs,postLayerDeltaA];CNBackPropogateLayer[FullyConnected1DTo1D[biases,Transpose[AutoencoderShared[[2]]]],postLayerDeltaA,_,_] )*)


(* ::Input:: *)
(*CNGradLayer[Tied[biases_],layerInputs_,layerOutputDelta_] := *)
(*(myg=CNGradLayer[FullyConnected1DTo1D[biases,Transpose[AutoencoderShared[[2]]]],layerInputs,layerOutputDelta][[1]]);*)


(* ::Input:: *)
(*CNLayerWeightPlus[Tied[biases_],grad_] := Tied[biases+grad];*)


(* ::Input:: *)
(*layerB=Table[0,{400}];layerW=Table[(Random[]-.5)/400,{400},{256}];*)


(* ::Input:: *)
(*outB=Table[0,{256}];*)


SeedRandom[1234];
autoencoder={
   Shared[layerB,layerW],Logistic,
   Tied[outB],Logistic
};


(* ::Input:: *)
(*tr= Map[Flatten[ImageData[#]]->Flatten[ImageData[#]]&,TrainingImages];*)


naturalTrain=Map[(Flatten[#]->Flatten[#])&,dat];


images=Import["C:\\Users\\julian\\ImageDataSets\\Distractors\\Distractors.wdx"];


patches=Flatten[Map[ImagePartition[#,{16,16}]&,images],2];


vecs=RandomSample[Map[Flatten[ImageData[#]]&,patches]];vecs//Dimensions


tr=Map[(#->#)&,vecs[[1;;10000]]];


addGaussian[trainingSet_] := 
   Map[
      Function[
         trainingItem,
         trainingItem[[1]] + RandomVariate[NormalDistribution[0,0.5],Length[trainingItem[[1]]]]->trainingItem[[2]]],
      trainingSet];


CurrentModel=autoencoder;


(* ::Input:: *)
(*CNMiniBatchTrainModel[ CurrentModel,tr,CNCrossEntropyLoss,{*)
(*   Momentum->0.9,MomentumType->"Nesterov",MaxEpoch->10^6,PreprocessTrainingSet->addGaussian,*)
(*   LearningRate->.001}]*)
(**)


Export["C:\\Users\\julian\\Google Drive\\Personal\\Computer Science\\WebMonitor\\Autoencoders\\DenoisingBinary.wdx",CurrentModel];
