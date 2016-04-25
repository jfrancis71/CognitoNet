(* ::Package:: *)

(*
   Layer: FullyConnected1DTo1D
*)
SyntaxInformation[FullyConnected1DTo1D]={"ArgumentsPattern"->{_,_}};
CNForwardPropogateLayer[FullyConnected1DTo1D[layerBiases_List,layerWeights_?MatrixQ],inputs_List]:=(
   CNAssertAbort[(layerWeights//First//Length)==(Transpose[inputs]//Length),
      "FullyConnected1DTo1D::Weight-Activation Error. Input length inconsistent with weight matrix."];
   CNAssertAbort[(layerBiases//Length)==(layerWeights//Length),
      "FullyConnected1DTo1D::Weight-Weight Error. Layer specification internally inconsistent."];
   Transpose[layerWeights.Transpose[inputs] + layerBiases] (* Note rows in inputs correspond to examples,
      and cols to neurons. + operator adds layerBiases columnwise to the matrix, hence these transposes. *)
);
CNLayerNumberParameters[FullyConnected1DTo1D[layerBiases_List,layerWeights_?MatrixQ]] :=
   Length[Flatten[layerWeights]] + Length[layerBiases];


(*
   Layer: Convolve2D
*)
SyntaxInformation[Convolve2D]={"ArgumentsPattern"->{_,_}};
CNForwardPropogateLayer[Convolve2D[layerBias_,layerKernel_],inputs_]:=(
   ListCorrelate[{layerKernel},inputs]+layerBias
);
CNLayerNumberParameters[Convolve2D[layerBias_,layerKernel_]] :=
   Length[Flatten[layerKernel]] + 1;


(*
   Layer: Convolve2DToFilterBank
*)
SyntaxInformation[Convolve2DToFilterBank]={"ArgumentsPattern"->{_}};
CNForwardPropogateLayer[Convolve2DToFilterBank[filters_],inputs_] := (
   CNAssertAbort[(inputs[[1]]//Dimensions//Length)==2,"Convolve2DToFilterBank::inputs does not match 2D structure"];
   Transpose[Map[CNForwardPropogateLayer[#,inputs]&,filters],{2,1,3,4}]
);
CNLayerNumberParameters[Convolve2DToFilterBank[filters_]] :=
   Total[Map[CNLayerNumberParameters,filters]];


(*
   Layer: ConvolveFilterBankTo2D
*)
SyntaxInformation[ConvolveFilterBankTo2D]={"ArgumentsPattern"->{_,_}};
CNForwardPropogateLayer[ConvolveFilterBankTo2D[bias_,kernels_],inputs_] := (
   CNAssertAbort[Length[inputs[[1]]]==Length[kernels],
      "ConvolveFilterBankTo2D::#Kernels ("<>ToString[Length[kernels]]<>") not equal to #Features ("<>ToString[Length[inputs[[1]]]]<>") in input feature map"];
   bias+Sum[ListCorrelate[{kernels[[kernel]]},inputs[[All,kernel]]],
      {kernel,1,Length[kernels]}]);
CNLayerNumberParameters[ConvolveFilterBankTo2D[bias_,kernels_]] :=
   Length[Flatten[kernels]] + 1;


(*
   Layer: ConvolveFilterBankToFilterBank
*)
SyntaxInformation[ConvolveFilterBankToFilterBank]={"ArgumentsPattern"->{_}};
(* Ref: http://www.jimmyren.com/papers/aaai_vcnn.pdf
   On Vectorization of deep convolutional neural networks for vision tasks
   Jimmy Ren, Li Xu, 2015
*)
CNForwardPropogateLayer[ConvolveFilterBankToFilterBank[filters_],inputs_]:=Module[{i1,i2,i3},
   i1 = Map[Partition[#,{5,5},{1,1}]&,inputs,{2}];
   i2=(Map[Flatten,
         Transpose[i1,{1,4,2,3,5,6}],{3}].
      Transpose[Map[Flatten,filters[[All,2]]]]);
   i3=Transpose[i2,{1,3,4,2}];
   Do[i3[[All,t]]=i3[[All,t]]+filters[[All,1]][[t]],{t,1,Length[i3[[1]]]}];i3
];
CNLayerNumberParameters[ConvolveFilterBankToFilterBank[filters_]] :=
   Total[Map[CNLayerNumberParameters,filters]]


(*
   Layer: Logistic
*)
CNLogisticFn[inputs_] := 1./(1.+Exp[-inputs])
SyntaxInformation[Logistic] = {"ArgumentsPattern"->{}};
CNForwardPropogateLayer[Logistic,inputs_] := CNLogisticFn[inputs];
CNLayerNumberParameters[Logistic] := 0;


(*
   Layer: Tanh
*)
CNForwardPropogateLayer[Tanh,inputs_]:=Tanh[inputs];
CNLayerNumberParameters[Tanh] := 0;


(*
   Layer: ReLU
*)
SyntaxInformation[ReLU]={"ArgumentsPattern"->{}};
CNForwardPropogateLayer[ReLU,inputs_]:=UnitStep[inputs-0]*inputs;


(*
   Layer: MaxPoolingFilterBankToFilterBank
*)
SyntaxInformation[MaxPoolingFilterBankToFilterBank]={"ArgumentsPattern"->{}};
CNForwardPropogateLayer[MaxPoolingFilterBankToFilterBank,inputs_] :=
   Map[Function[image,Map[Max,Partition[image,{2,2}],{2}]],inputs,{2}];
CNLayerNumberParameters[MaxPoolingFilterBankToFilterBank] := 0;


(*
   Layer: MaxConvolveFilterBankToFilterBank
*)
SyntaxInformation[MaxConvolveFilterBankToFilterBank]={"ArgumentsPattern"->{}};
CNForwardPropogateLayer[MaxConvolveFilterBankToFilterBank,inputs_]:=
   Map[Max,Map[Partition[#,{3,3},{1,1},{-2,+2},-2.0]&,inputs,{2}],{4}];
CNLayerNumberParameters[MaxConvolveFilterBankToFilterBank] := 0;


(*
   Layer: Softmax
*)
SyntaxInformation[Softmax]={"ArgumentsPattern"->{}};
CNForwardPropogateLayer[Softmax,inputs_] := Map[Exp[#]/Total[Exp[#]]&,inputs];
CNLayerNumberParameters[Softmax] := 0;


(*
   Layer: AdaptorFilterBankTo1D
*)
(* Helper Function sources from Mathematica on-line documentation regarding example use of Partition *)
unflatten[e_,{d__?((IntegerQ[#]&&Positive[#])&)}]:= 
   Fold[Partition,e,Take[{d},{-1,2,-1}]] /;(Length[e]===Times[d]);
SyntaxInformation[AdaptorFilterBankTo1D]={"ArgumentsPattern"->{_,_,_}};
CNForwardPropogateLayer[AdaptorFilterBankTo1D[features_,width_,height_],inputs_]:=(
   Map[Flatten,inputs]
);
CNLayerNumberParameters[AdaptorFilterBankTo1D[features_,width_,height_]] := 0;

(* Deprecated *)
CNForwardPropogateLayer[Adaptor3DTo1D[features_,width_,height_],inputs_]:=(
   Map[Flatten,inputs]
);
CNLayerNumberParameters[Adaptor3DTo1D[features_,width_,height_]] := 0;


(*
   Layer: PadFilterBank
*)
SyntaxInformation[PadFilterBank] = {"ArgumentsPattern"->{_}};
CNForwardPropogateLayer[PadFilterBank[padding_],inputs_] := Map[ArrayPad[#,padding,.0]&,inputs,{2}];
CNLayerNumberParameters[PadFilterBank[padding_]] := 0;


(*
   Layer: SubsampleFilterBankToFilterBank
*)
SyntaxInformation[SubsampleFilterBankToFilterBank] = {"ArgumentsPattern"->{}};
CNForwardPropogateLayer[SubsampleFilterBankToFilterBank,inputs_] := Map[#[[1;;-1;;2,1;;-1;;2]]&,inputs,{2}];
CNLayerNumberParameters[SubsampleFilterBankToFilterBank] := 0;


(*
   Layer: PadFilter
*)
SyntaxInformation[PadFilter]={"ArgumentsPattern"->{_}};
CNForwardPropogateLayer[PadFilter[padding_],inputs_]:=Map[ArrayPad[#,padding,.0]&,inputs];
CNLayerNumberParameters[PadFilter[padding_]] := 0;


(*
   Ref: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
*)
SyntaxInformation[DropoutLayer]={"ArgumentsPattern"->{_,_}};
SyntaxInformation[DropoutLayerMask]={"ArgumentsPattern"->{_}};
Dropout[layer_,inputs_]:=layer;
Dropout[net_List,inputs_]:=Map[Dropout[#,inputs]&,net];
Dropout[DropoutLayer[dims_,dropoutProb_],inputs_]:=
   DropoutLayerMask[Table[RandomInteger[],{Length[inputs]},dims]];
CNForwardPropogateLayer[DropoutLayer[_,_],inputs_]:=0.5*inputs;
CNForwardPropogateLayer[DropoutLayerMask[mask_],inputs_]:=inputs*mask;
CNLayerNumberParameters[DropoutLayer[_,_]] := 0;


(*
   Default Description
*)
CNLayerDescription[layer_Symbol]:=ToString[layer];
CNLayerDescription[layer_]:=ToString[Head[layer]];
