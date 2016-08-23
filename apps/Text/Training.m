(* ::Package:: *)

(* ::Input:: *)
(*files=FileNames["c:\\users\\julian\\imagedatasets\\historytext\\*.jpg"];*)


(* ::Input:: *)
(*dat=Map[ColorConvert[ImageResize[ImageRotate[Import[#]],Scaled[0.5]],"GrayScale"]&,files[[1;;-1]]];*)


(* ::Input:: *)
(*frags=RandomSample[Flatten[Map[ImagePartition[#,{64,64}]&,dat]]];*)


(* ::Input:: *)
(*trainingSet=Map[*)
(*If[Random[]>.5,*)
(*ImageReflect[#,Left->Right]->1,*)
(*#->0]&,frags[[1;;14000]]];*)


(* ::Input:: *)
(*validationSet=Map[*)
(*If[Random[]>.5,*)
(*ImageReflect[#,Left->Right]->1,*)
(*#->0]&,frags[[14001;;-1]]];*)


(* ::Input:: *)
(*Export["c:\\users\\julian\\imagedatasets\\historytext\\dataset.mx",{trainingSet,validationSet}];*)
