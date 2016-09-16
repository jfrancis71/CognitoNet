(* ::Package:: *)

(* ::Input:: *)
(*<<CNNeuralCore.m*)


(* ::Input:: *)
(*SocketBaseDir = "C:\\Users\\julian\\ImageDataSets\\sockets";*)


(* ::Input:: *)
(*strainingPositives=Map[*)
(*   Function[fileName,*)
(*      Map[ImageTake[#,{71-32,71+31},{137-32,137+31}]&,CNImportMovie[fileName,256]]],*)
(*   FileNames[SocketBaseDir<>"\\Training\\*"]];*)


(* ::Input:: *)
(*svalidationPositives=Map[*)
(*   Function[fileName,*)
(*      Map[ImageTake[#,{71-32,71+31},{137-32,137+31}]&,CNImportMovie[fileName,256]]],*)
(*   FileNames[SocketBaseDir<>"\\Validation\\*"]];*)


noEntryBaseDir = "C:\\Users\\julian\\ImageDataSets\\NoEntrySigns5";


strainingNegatives=Map[
   Function[fileName,
      Map[ImagePartition[#,{64,64}]&,CNImportMovie[fileName,256]]],
   FileNames[noEntryBaseDir<>"\\Training\\Negatives\\*.avi"]];


svalidationNegatives=Map[
   Function[fileName,
      Map[ImagePartition[#,{64,64}]&,CNImportMovie[fileName,256]]],
   FileNames[noEntryBaseDir<>"\\Validation\\Negatives\\*"]];


strainingSet=RandomSample[Join[
   Map[#->1&,Flatten[strainingPositives]],
   Map[#->0&,(Flatten[strainingNegatives]//RandomSample)[[1;;Length[Flatten[strainingPositives]]]]]
]];


svalidationSet=RandomSample[Join[
   Map[#->1&,Flatten[svalidationPositives]],
   Map[#->0&,(Flatten[svalidationNegatives]//RandomSample)[[1;;Length[Flatten[svalidationPositives]]]]]
]];


(* ::Input:: *)
(*(*Export["C:\\Users\\julian\\ImageDataSets\\sockets\\DataSet.mx",{trainingSet,validationSet}];*)*)


(*Export["C:\\Users\\julian\\ImageDataSets\\sockets\\DataSetv1.mx",{trainingSet,validationSet}];*)


(* ::Input:: *)
(*SeedRandom[1234];*)
(*SocketNet={*)
(*   Convolve2DToFilterBankInit[16,5],Tanh,*)
(*   MaxConvolveFilterBankToFilterBank,SubsampleFilterBankToFilterBank,ConvolveFilterBankToFilterBankInit[16,16,5],Tanh,*)
(*   MaxConvolveFilterBankToFilterBank,SubsampleFilterBankToFilterBank,ConvolveFilterBankToFilterBankInit[16,32,5],Tanh,*)
(*   ConvolveFilterBankTo2DInit[32,1],*)
(*   Adaptor2DTo1D[9],*)
(*   LogSumExp,*)
(*   Logistic*)
(*};*)


checkpointDir = "Sockets\\clean\\SocketNet";


CNMiniBatchTrainModel[ CNConvertCPUToGPU[SocketNet], trainingSet,CNCrossEntropyLoss,{
   ValidationSet->validationSet,Momentum->0.9,MomentumType->"Nesterov",
   LearningRate->.01,
   EpochMonitor:>CNCheckpoint[checkpointDir]}]

