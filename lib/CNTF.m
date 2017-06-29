(* ::Package:: *)

(*
   CognitoNet code to convert CognitoNet neural layers to the TensorFlow format:
      NHWC
*)


(* Warning running python Tkinter code on a Mac seems to interfere with the Mathematica
   front end on the Mac, so you might not want to run at the same time. Causes screen locks
*)


CNConvertToTFConvolve2DToFilterBankWeights[layer_]:=
   {layer[[1,All,1]],Transpose[{layer[[1,All,2]]},{3,4,1,2}]}


CNConvertToTFConvolveFilterBankToFilterBankWeights[layer_]:=
   {layer[[1,All,1]],Transpose[layer[[1,All,2]],{4,3,1,2}]}


CNConvertToTFConvolveFilterBankTo2DWeights[layer_]:=
   {layer[[1]],Transpose[{layer[[2]]},{4,3,1,2}]}


CNConvertToTFFullyConnected1DToScalarWeights[layer_] :=
   {layer[[1]], Transpose[{layer[[2]]}]}
