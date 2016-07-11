(* ::Package:: *)

<<CNNeuralCore.m


(*
Set noEntryBaseDir and checkpointDir to relevent directories, eg

   noEntryBaseDir = "C:\\Users\\julian\\ImageDataSets\\NoEntrySigns5";
   checkpointDir = "NoEntry5\\History1\\NoEntryNet";
*)


noEntryBaseDir = "C:\\Users\\julian\\ImageDataSets\\NoEntrySigns5";


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





(* Synthetic Code *)


randReflect[image_]:=If[Random[]>.5,ImageReflect[image],image]


transform[image_]:=
randReflect[ImageResize[ImagePad[
ImageRotate[image,(Random[]-.5)/10,{64,64}]
,{{Random[]*4,Random[]*4},{Random[]*4,Random[]*4}}],{64,64}]]


Length[trainingPositives//Flatten]


16567


Length[trainingNegatives//Flatten]


22680


Length[trainingNegatives//Flatten]


93888


Length[validationPositives//Flatten]


2971


Length[validationNegatives//Flatten]


65128


syntheticTrainingPositives=
Map[transform,Flatten[Table[trainingPositives,{10}]]];


syntheticTrainingNegatives=
Map[transform,Flatten[Table[trainingNegatives,{10}]]];


syntheticValidationPositives=
Map[transform,Flatten[Table[validationPositives,{10}]]];


syntheticValidationNegatives=
Map[transform,Flatten[Table[validationNegatives,{10}]]];


trainingSet=RandomSample[Join[
   Map[#->1&,Flatten[syntheticTrainingPositives]],
   Map[#->0&,Flatten[syntheticTrainingNegatives//RandomSample][[1;;Length[Flatten[syntheticTrainingPositives]]]]]
]];


validationSet=RandomSample[Join[
   Map[#->1&,Flatten[syntheticValidationPositives]],
   Map[#->0&,Flatten[syntheticValidationNegatives//RandomSample][[1;;Length[Flatten[syntheticValidationPositives]]]]]
]];


Length[trainingSet]


331340


Length[validationSet]


59420


(*Export["C:\\Users\\julian\\ImageDataSets\\NoEntrySigns5\\DataSetSynthetic.mx",{trainingSet,validationSet}];*)
