(* ::Package:: *)

<<CNNeuralCore.m


(*
Set noEntryBaseDir and checkpointDir to relevent directories, eg

   noEntryBaseDir = "C:\\Users\\julian\\ImageDataSets\\NoEntrySigns5";
   checkpointDir = "NoEntry5\\History1\\NoEntryNet";
*)


trainingPositives=Map[
   Function[fileName,
      Map[ImageTake[#,{71-32,71+31},{137-32,137+31}]&,CNImportMovie[fileName,256]]],
   FileNames[noEntryBaseDir<>"\\Training\\Positives\\*"]];


validationPositives=Map[
   Function[fileName,
      Map[ImageTake[#,{71-32,71+31},{137-32,137+31}]&,CNImportMovie[fileName,256]]],
   FileNames[noEntryBaseDir<>"\\Validation\\Positives\\*"]];


trainingNegatives=Map[
   Function[fileName,
      Map[ImagePartition[#,{64,64}]&,CNImportMovie[fileName,256]]],
   FileNames[noEntryBaseDir<>"\\Training\\Negatives\\*"]];


validationNegatives=Map[
   Function[fileName,
      Map[ImagePartition[#,{64,64}]&,CNImportMovie[fileName,256]]],
   FileNames[noEntryBaseDir<>"\\Validation\\Negatives\\*"]];


trainingSet=RandomSample[Join[
   Map[#->1&,Flatten[trainingPositives]],
   Map[#->0&,Flatten[trainingNegatives//RandomSample][[1;;Length[Flatten[trainingPositives]]]]]
]];


validationSet=RandomSample[Join[
   Map[#->1&,Flatten[validationPositives]],
   Map[#->0&,Flatten[validationNegatives//RandomSample][[1;;Length[Flatten[validationPositives]]]]]
]];


SeedRandom[1234];
NoEntryNet={
   Convolve2DToFilterBankInit[16,5],Tanh,
   MaxConvolveFilterBankToFilterBank,SubsampleFilterBankToFilterBank,ConvolveFilterBankToFilterBankInit[16,16,5],Tanh,
   MaxConvolveFilterBankToFilterBank,SubsampleFilterBankToFilterBank,ConvolveFilterBankToFilterBankInit[16,32,5],Tanh,
   ConvolveFilterBankTo2DInit[32,1],
   Adaptor2DTo1D[9],
   LogSumExp,
   Logistic
};


(*{trainingSet,validationSet}=Import["C:\\Users\\julian\\ImageDataSets\\NoEntrySigns4\\DataSet.mx"];*)


checkpointDir = "NoEntry5\\History3\\NoEntryNet";


CNMiniBatchTrainModel[ CNConvertCPUToGPU[NoEntryNet], trainingSet,CNCrossEntropyLoss,{
   ValidationSet->validationSet,Momentum->0.9,MomentumType->"Nesterov",
   LearningRate->.01,
   EpochMonitor:>CNCheckpoint[checkpointDir]}]



