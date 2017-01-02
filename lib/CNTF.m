(* ::Package:: *)

(*
   CognitoNet code to convert CognitoNet neural layers to the TensorFlow format:
      NHWC
*)


CNConvertToTFConvolve2DToFilterBankWeights[layer_]:=
   Transpose[{layer[[1,All,2]]},{3,4,1,2}]


CNConvertToTFConvolve2DToFilterBankBiases[layer_]:=
   layer[[1,All,1]]


CNConvertToTFConvolveFilterBankToFilterBankWeights[layer_]:=
   Transpose[layer[[1,All,2]],{4,3,1,2}]


CNConvertToTFConvolveFilterBankToFilterBankBiases[layer_]:=
   layer[[1,All,1]]


CNConvertToTFConvolveFilterBankTo2D[layer_]:=
   Transpose[{layer[[2]]},{4,3,1,2}]
