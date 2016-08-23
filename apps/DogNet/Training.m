(* ::Package:: *)

(* ::Input:: *)
(*<<CNNeuralCore.m*)


(* ::Input:: *)
(*pad[image_]:=({h,v}={64,64}-ImageDimensions[image];ImagePad[image,{{Floor[h/2],Ceiling[h/2]},{Floor[v/2],Ceiling[v/2]}}])*)


(* ::Input:: *)
(*dogTrainFiles=FileNames["C:\\Users\\julian\\ImageDataSets\\dogsandcats\\train\\dog*"];*)


(* ::Input:: *)
(*dogsTrain=Map[ColorConvert[pad[ImageResize[Import[#],{64}]],"GrayScale"]&,dogTrainFiles];*)


(* ::Input:: *)
(*catTrainFiles=FileNames["C:\\Users\\julian\\ImageDataSets\\dogsandcats\\train\\cat*"];*)


(* ::Input:: *)
(*catsTrain=Map[ColorConvert[pad[ImageResize[Import[#],{64}]],"GrayScale"]&,catTrainFiles];*)


(* ::Input:: *)
(*dogValidFiles=FileNames["C:\\Users\\julian\\ImageDataSets\\dogsandcats\\valid\\dog*"];*)


(* ::Input:: *)
(*dogsValid=Map[ColorConvert[pad[ImageResize[Import[#],{64}]],"GrayScale"]&,dogValidFiles];*)


(* ::Input:: *)
(*catValidFiles=FileNames["C:\\Users\\julian\\ImageDataSets\\dogsandcats\\valid\\cat*"];*)


(* ::Input:: *)
(*catsValid=Map[ColorConvert[pad[ImageResize[Import[#],{64}]],"GrayScale"]&,catValidFiles];*)


(* ::Input:: *)
(*Length[dogsTrain]*)


(* ::Input:: *)
(*Length[catsTrain]*)


(* ::Input:: *)
(*Length[dogTrainFiles]*)


(* ::Input:: *)
(*Length[dogsValid]*)


(* ::Input:: *)
(*Length[catsValid]*)


(* ::Input:: *)
(*trainingSet=RandomSample[Join[*)
(*   Map[#->1&,dogsTrain],*)
(*   Map[#->0&,catsTrain]*)
(*]];*)


(* ::Input:: *)
(*validationSet=RandomSample[Join[*)
(*   Map[#->1&,dogsValid],*)
(*   Map[#->0&,catsValid]*)
(*]];*)


(* ::Input:: *)
(*Length[dogsTrain]*)


(* ::Input:: *)
(*Map[ImageDimensions,dogsTrain]*)


(* ::Input:: *)
(*dogs*)


(* ::Input:: *)
(*<<CNGPULayers.m*)


(* ::Input:: *)
(*CNGPUInitialize[]*)


(* ::Input:: *)
(*SeedRandom[1234];*)
(*DogNet={*)
(*   Convolve2DToFilterBankInit[16,5],Tanh,*)
(*   MaxConvolveFilterBankToFilterBank,SubsampleFilterBankToFilterBank,ConvolveFilterBankToFilterBankInit[16,16,5],Tanh,*)
(*   MaxConvolveFilterBankToFilterBank,SubsampleFilterBankToFilterBank,ConvolveFilterBankToFilterBankInit[16,32,5],Tanh,*)
(*   ConvolveFilterBankToFilterBankInit[32,32,5],*)
(*   AdaptorFilterBankTo1D[32,5,5],*)
(*   FullyConnected1DToScalarInit[800],*)
(*   Logistic*)
(*};*)


(* ::Input:: *)
(*CNMiniBatchTrainModel[ CNConvertCPUToGPU[DogNet], trainingSet,CNCrossEntropyLoss,{*)
(*   ValidationSet->validationSet,Momentum->0.9,MomentumType->"Nesterov",*)
(*   LearningRate->.01,*)
(*   EpochMonitor:>CNCheckpoint["DogNet\\History1"]}]*)
(**)
