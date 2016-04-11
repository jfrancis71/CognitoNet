(* ::Package:: *)

(* NOTE TO ME Should Transpose below be other way round? *)


(*
   Layer: FullyConnected1DTo1D
*)
SyntaxInformation[FullyConnected1DTo1D]={"ArgumentsPattern"->{_,_}};
ForwardPropogateLayer[FullyConnected1DTo1D[layerBiases_List,layerWeights_?MatrixQ],inputs_List]:=(
   CNAssertAbort[(layerWeights//First//Length)==(Transpose[inputs]//Length),
      "FullyConnected1DTo1D::Weight-Activation Error. Input length inconsistent with weight matrix."];
   CNAssertAbort[(layerBiases//Length)==(layerWeights//Length),
      "FullyConnected1DTo1D::Weight-Weight Error. Layer specification internally inconsistent."];
   Transpose[layerWeights.Transpose[inputs] + layerBiases]
);


(*
   Layer: Convolve2D
*)
SyntaxInformation[Convolve2D]={"ArgumentsPattern"->{_,_}};
ForwardPropogateLayer[Convolve2D[layerBias_,layerKernel_],inputs_]:=(
   ListCorrelate[{layerKernel},inputs]+layerBias
);


(*
   Layer: Convolve2DToFilterBank
*)
SyntaxInformation[Convolve2DToFilterBank]={"ArgumentsPattern"->{_}};
ForwardPropogateLayer[Convolve2DToFilterBank[filters_],inputs_]:=(
   CNAssertAbort[(inputs[[1]]//Dimensions//Length)==2,"Convolve2DToFilterBank::inputs does not match 2D structure"];
   Transpose[Map[ForwardPropogateLayer[#,inputs]&,filters],{2,1,3,4}]
);


(*
   Layer: ConvolveFilterBankTo2D
*)
SyntaxInformation[ConvolveFilterBankTo2D]={"ArgumentsPattern"->{_,_}};
ForwardPropogateLayer[ConvolveFilterBankTo2D[bias_,kernels_],inputs_]:=(
   CNAssertAbort[Length[inputs[[1]]]==Length[kernels],
      "ConvolveFilterBankTo2D::#Kernels ("<>ToString[Length[kernels]]<>") not equal to #Features ("<>ToString[Length[inputs[[1]]]]<>") in input feature map"];
   bias+Sum[ListCorrelate[{kernels[[kernel]]},inputs[[All,kernel]]],
      {kernel,1,Length[kernels]}]);


(*
   Layer: ConvolveFilterBankToFilterBank
*)
SyntaxInformation[ConvolveFilterBankToFilterBank]={"ArgumentsPattern"->{_}};
(* Ref: http://www.jimmyren.com/papers/aaai_vcnn.pdf
   On Vectorization of deep convolutional neural networks for vision tasks
   Jimmy Ren, Li Xu, 2015
*)
ForwardPropogateLayer[ConvolveFilterBankToFilterBank[filters_],inputs_]:=Module[{i1,i2,i3},
   i1 = Map[Partition[#,{5,5},{1,1}]&,inputs,{2}];
   i2=(Map[Flatten,
         Transpose[i1,{1,4,2,3,5,6}],{3}].
      Transpose[Map[Flatten,filters[[All,2]]]]);
   i3=Transpose[i2,{1,3,4,2}];
   Do[i3[[All,t]]=i3[[All,t]]+filters[[All,1]][[t]],{t,1,Length[i3[[1]]]}];i3
];


(*
   Layer: Tanh
*)
ForwardPropogateLayer[Tanh,inputs_]:=Tanh[inputs];


(*
   Layer: MaxPoolingFilterBankToFilterBank
*)
SyntaxInformation[MaxPoolingFilterBankToFilterBank]={"ArgumentsPattern"->{}};
ForwardPropogateLayer[MaxPoolingFilterBankToFilterBank,inputs_]:=
   Map[Function[image,Map[Max,Partition[image,{2,2}],{2}]],inputs,{2}];



(*
   Layer: MaxConvolveFilterBankToFilterBank
*)
SyntaxInformation[MaxConvolveFilterBankToFilterBank]={"ArgumentsPattern"->{}};
ForwardPropogateLayer[inputs_,MaxConvolveFilterBankToFilterBank]:=
   Map[Max,Map[Partition[#,{3,3},{1,1},{-2,+2},-2.0]&,inputs,{2}],{4}];


(*
   Layer: Adaptor3DTo1D
*)
(* Helper Function sources from Mathematica on-line documentation regarding example use of Partition *)
unflatten[e_,{d__?((IntegerQ[#]&&Positive[#])&)}]:= 
   Fold[Partition,e,Take[{d},{-1,2,-1}]] /;(Length[e]===Times[d]);
SyntaxInformation[Adaptor3DTo1D]={"ArgumentsPattern"->{_,_,_}};
ForwardPropogateLayer[Adaptor3DTo1D[features_,width_,height_],inputs_]:=(
   Map[Flatten,inputs]
);


(*
   Layer: Softmax
*)
SyntaxInformation[Softmax]={"ArgumentsPattern"->{}};
ForwardPropogateLayer[Softmax,inputs_]:=Map[Exp[#]/Total[Exp[#]]&,inputs];

