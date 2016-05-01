(* ::Package:: *)

(*
   Layer: FullyConnected1DToScalar
*)
SyntaxInformation[FullyConnected1DToScalar]={"ArgumentsPattern"->{_,_}};
FullyConnected1DToScalarInit[noFromNeurons_]:=
   FullyConnected1DToScalar[.0,Table[Random[]-.5,{noFromNeurons}]/Sqrt[noFromNeurons]]
CNForwardPropogateLayer[FullyConnected1DToScalar[layerBias_,layerWeights_?VectorQ],inputs_List]:=(
   inputs.layerWeights + layerBias (* Note rows in inputs correspond to examples,
      and cols to neurons. + operator adds layerBiases columnwise to the matrix, hence these transposes. *)
);
CNBackPropogateLayer[FullyConnected1DToScalar[bias_,weights_],postLayerDeltaA_,_,_]:=Transpose[{postLayerDeltaA}].{weights};
CNGradLayer[FullyConnected1DToScalar[bias_,weights_],layerInputs_,layerOutputDelta_]:={Total[layerOutputDelta],Transpose[layerInputs].layerOutputDelta};
CNLayerWeightPlus[networkLayer_FullyConnected1DToScalar,grad_]:=FullyConnected1DToScalar[networkLayer[[1]]+grad[[1]],networkLayer[[2]]+grad[[2]]];
CNLayerNumberParameters[FullyConnected1DToScalar[layerBias_,layerWeights_?VectorQ]] :=
   Length[layerWeights] + 1;


(*
   Layer: FullyConnected1DTo1D
*)
SyntaxInformation[FullyConnected1DTo1D]={"ArgumentsPattern"->{_,_}};
FullyConnected1DTo1DInit[noFromNeurons_,noToNeurones_]:=
   FullyConnected1DTo1D[ConstantArray[0.,noToNeurones],Table[Random[]-.5,{noToNeurones},{noFromNeurons}]/Sqrt[noFromNeurons]]
CNForwardPropogateLayer[FullyConnected1DTo1D[layerBiases_List,layerWeights_?MatrixQ],inputs_List]:=(
   CNAssertAbort[(layerWeights//First//Length)==(Transpose[inputs]//Length),
      "FullyConnected1DTo1D::Weight-Activation Error. Input length inconsistent with weight matrix."];
   CNAssertAbort[(layerBiases//Length)==(layerWeights//Length),
      "FullyConnected1DTo1D::Weight-Weight Error. Layer specification internally inconsistent."];
   Transpose[layerWeights.Transpose[inputs] + layerBiases] (* Note rows in inputs correspond to examples,
      and cols to neurons. + operator adds layerBiases columnwise to the matrix, hence these transposes. *)
);
CNBackPropogateLayer[FullyConnected1DTo1D[biases_,weights_],postLayerDeltaA_,_,_]:=postLayerDeltaA.weights
CNGradLayer[FullyConnected1DTo1D[biases_,weights_],layerInputs_,layerOutputDelta_]:={Total[Transpose[layerOutputDelta],{2}],Transpose[layerOutputDelta].layerInputs};
CNLayerWeightPlus[networkLayer_FullyConnected1DTo1D,grad_]:=FullyConnected1DTo1D[networkLayer[[1]]+grad[[1]],networkLayer[[2]]+grad[[2]]];
CNLayerNumberParameters[FullyConnected1DTo1D[layerBiases_List,layerWeights_?MatrixQ]] :=
   Length[Flatten[layerWeights]] + Length[layerBiases];


(*
   Layer: Convolve2D
*)
SyntaxInformation[Convolve2D]={"ArgumentsPattern"->{_,_}};
CNForwardPropogateLayer[Convolve2D[layerBias_,layerKernel_],inputs_]:=(
   ListCorrelate[{layerKernel},inputs]+layerBias
);
CNBackPropogateLayer[Convolve2D[biases_,weights_],postLayerDeltaA_,_,_]:=Table[ListConvolve[weights,postLayerDeltaA[[t]],{+1,-1},0],{t,1,Length[postLayerDeltaA]}];
CNGradLayer[Convolve2D[biases_,weights_],layerInputs_,layerOutputDelta_]:={Total[layerOutputDelta,3],Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta,layerInputs}]]};
CNLayerWeightPlus[networkLayer_Convolve2D,grad_]:=Convolve2D[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]];
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
CNBackPropogateLayer[Convolve2DToFilterBank[filters_],postLayerDeltaA_,inputs_,outputs_]:=Sum[CNBackPropogateLayer[filters[[f]],postLayerDeltaA[[All,f]],inputs,outputs],{f,1,Length[filters]}]
CNGradLayer[Convolve2DToFilterBank[filters_],layerInputs_,layerOutputDelta_]:=
   Table[{
         Total[layerOutputDelta[[All,filterIndex]],3],
         Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta[[All,filterIndex]],layerInputs}]]},
      {filterIndex,1,Length[filters]}];
CNLayerWeightPlus[networkLayer_Convolve2DToFilterBank,grad_]:=Convolve2DToFilterBank[CNLayerWeightPlus[networkLayer[[1]],grad]];
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
CNBackPropogateLayer[ConvolveFilterBankTo2D[bias_,kernels_],postLayerDeltaA_,_,_]:=(
   Transpose[Table[ListConvolve[{kernels[[w]]},postLayerDeltaA,{+1,-1},0],{w,1,Length[kernels]}],{2,1,3,4}]);
CNGradLayer[ConvolveFilterBankTo2D[bias_,kernels_],layerInputs_,layerOutputDelta_]:=(
   (*{Total[layerOutputDelta,3],Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta,layerInputs}]]}*)
   {Total[layerOutputDelta,3],Table[Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta,layerInputs[[All,w]]}]],{w,1,Length[kernels]}]});
CNLayerWeightPlus[networkLayer_ConvolveFilterBankTo2D,grad_]:=ConvolveFilterBankTo2D[networkLayer[[1]]-grad[[1]],networkLayer[[2]]-grad[[2]]];
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
CNBackPropogateLayer[ConvolveFilterBankToFilterBank[filters_],postLayerDeltaA_,inputs_,outputs_]:=
   Sum[CNBackPropogateLayer[filters[[f]],postLayerDeltaA[[All,f]],inputs,outputs],{f,1,Length[filters]}];
CNGradLayer[ConvolveFilterBankToFilterBank[filters_],layerInputs_,layerOutputDelta_]:=
   Table[{
      Total[layerOutputDelta[[All,filterOutputIndex]],3],
      ListCorrelate[Transpose[{layerOutputDelta[[All,filterOutputIndex]]},{2,1,3,4}],layerInputs][[1]]},
      {filterOutputIndex,1,Length[filters]}]
CNLayerWeightPlus[ConvolveFilterBankToFilterBank[filters_],grad_]:=ConvolveFilterBankToFilterBank[WeightDec[filters,grad]];
CNLayerNumberParameters[ConvolveFilterBankToFilterBank[filters_]] :=
   Total[Map[CNLayerNumberParameters,filters]]


(*
   Layer: Logistic
*)
CNLogisticFn[inputs_] := 1./(1.+Exp[-inputs])
SyntaxInformation[Logistic] = {"ArgumentsPattern"->{}};
CNForwardPropogateLayer[Logistic,inputs_] := CNLogisticFn[inputs];
CNBackPropogateLayer[Logistic,postLayerDeltaA_,inputs_,_]:=
   postLayerDeltaA*Exp[inputs]*(1+Exp[inputs])^-2;
CNGradLayer[Logistic,layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[Logistic,grad_]:=Logistic;
CNLayerNumberParameters[Logistic] := 0;


(*
   Layer: Tanh
*)
CNForwardPropogateLayer[Tanh,inputs_]:=Tanh[inputs];
CNBackPropogateLayer[Tanh,postLayerDeltaA_,inputs_,_]:=
   postLayerDeltaA*Sech[inputs]^2;
CNGradLayer[Tanh,layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[Tanh,grad_]:=Tanh;
CNLayerNumberParameters[Tanh] := 0;


(*
   Layer: ReLU
*)
SyntaxInformation[ReLU]={"ArgumentsPattern"->{}};
CNForwardPropogateLayer[ReLU,inputs_]:=UnitStep[inputs-0]*inputs;
CNBackPropogateLayer[ReLU,postLayerDeltaA_,inputs_,_]:=
   postLayerDeltaA*UnitStep[inputs-0];
CNGradLayer[ReLU,layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[ReLU,grad_]:=ReLU;


(*
   Layer: MaxPoolingFilterBankToFilterBank
*)
SyntaxInformation[MaxPoolingFilterBankToFilterBank]={"ArgumentsPattern"->{}};
CNForwardPropogateLayer[MaxPoolingFilterBankToFilterBank,inputs_] :=
   Map[Function[image,Map[Max,Partition[image,{2,2}],{2}]],inputs,{2}];
UpSample[x_]:=Riffle[temp=Riffle[x,x]//Transpose;temp,temp]//Transpose;
backRouting[previousZ_,nextA_]:=UnitStep[previousZ-Map[UpSample,nextA,{2}]];
CNBackPropogateLayer[MaxPoolingFilterBankToFilterBank,postLayerDeltaA_,layerInputs_,layerOutputs_]:=
   backRouting[layerInputs,layerOutputs]*Map[UpSample,postLayerDeltaA,{2}];
CNGradLayer[MaxPoolingFilterBankToFilterBank,layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[MaxPoolingFilterBankToFilterBank,grad_]:=MaxPoolingFilterBankToFilterBank;
CNLayerNumberParameters[MaxPoolingFilterBankToFilterBank] := 0;


(*
   Layer: MaxConvolveFilterBankToFilterBank
*)
SyntaxInformation[MaxConvolveFilterBankToFilterBank]={"ArgumentsPattern"->{}};
CNForwardPropogateLayer[MaxConvolveFilterBankToFilterBank,inputs_]:=
   Map[Max,Map[Partition[#,{3,3},{1,1},{-2,+2},-2.0]&,inputs,{2}],{4}];
CNBackPropogateLayer[MaxConvolveFilterBankToFilterBank,postLayerDeltaA_,inputs_,outputs_]:=(
   CNAssertAbort[Max[inputs]<1.4,"BackPropogateLayer::MaxConvolveFilterBankToFilterBank algo not designed for inputs > 1.4"];
(*   u1=Map[Partition[#,{3,3},{1,1},{-2,+2},-2.0]&,inputs,{2}];
   u2=Map[Max[Flatten[#]]&,u1,{4}];*)
   CNTimer["MaxConvolveFilterBankToFilterBank::u3",u3=ToPackedArray[Map[Partition[#,{3,3},{1,1},{-2,+2},1.5]&,outputs,{2}]]];
   CNTimer["MaxConvolveFilterBankToFilterBank::u4",u4=UnitStep[inputs-u3]];
   CNTimer["MaxConvolveFilterBankToFilterBank::u5",u5=ToPackedArray[Map[Partition[#,{3,3},{1,1},{-2,+2},-2.0]&,postLayerDeltaA,{2}]]];
   CNTimer["MaxConvolveFilterBankToFilterBank::u6",u6=u4*u5];
   CNTimer["MaxConvolveFilterBankToFilterBank::u7",u7=Map[Total[Flatten[#]]&,u6,{4}]])
CNGradLayer[MaxConvolveFilterBankToFilterBank,layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[MaxConvolveFilterBankToFilterBank,grad_]:=MaxConvolveFilterBankToFilterBank;
CNLayerNumberParameters[MaxConvolveFilterBankToFilterBank] := 0;


(*
   Layer: Softmax
*)
SyntaxInformation[Softmax]={"ArgumentsPattern"->{}};
CNForwardPropogateLayer[Softmax,inputs_] := Map[Exp[#]/Total[Exp[#]]&,inputs];
CNBackPropogateLayer[Softmax,postLayerDeltaA_,_,outputs_]:=
   Table[
      Sum[postLayerDeltaA[[n,i]]*outputs[[n,i]]*(KroneckerDelta[j,i]-outputs[[n,j]]),{i,1,Length[postLayerDeltaA[[1]]]}],
         {n,1,Length[postLayerDeltaA]},
      {j,1,Length[postLayerDeltaA[[1]]]}];
CNGradLayer[Softmax,layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[Softmax,grad_]:=Softmax;
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
CNBackPropogateLayer[AdaptorFilterBankTo1D[features_,width_,height_],postLayerDeltaA_,_,_]:=
   unflatten[Flatten[postLayerDeltaA],{Length[postLayerDeltaA],features,width,height}];
CNGradLayer[AdaptorFilterBankTo1D[features_,width_,height_],layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[networkLayer_Adaptor3DTo1D,grad_]:=AdaptorFilterBankTo1D[networkLayer[[1]],networkLayer[[2]],networkLayer[[3]]];
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
CNBackPropogateLayer[PadFilterBank[padding_],postLayerDeltaA_,_,_]:=
   postLayerDeltaA[[All,All,padding+1;;-padding-1,padding+1;;-padding-1]];
CNGradLayer[PadFilterBank[padding_],layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[PadFilterBank[padding_],grad_]:=PadFilterBank[padding];
CNLayerNumberParameters[PadFilterBank[padding_]] := 0;


(*
   Layer: SubsampleFilterBankToFilterBank
*)
SyntaxInformation[SubsampleFilterBankToFilterBank] = {"ArgumentsPattern"->{}};
CNForwardPropogateLayer[SubsampleFilterBankToFilterBank,inputs_] := Map[#[[1;;-1;;2,1;;-1;;2]]&,inputs,{2}];
UpSample1[x_]:=Riffle[temp=Riffle[x,.0*x]//Transpose;temp,temp*.0]//Transpose;
CNBackPropogateLayer[SubsampleFilterBankToFilterBank,postLayerDeltaA_,_,_]:=
   Map[UpSample1,postLayerDeltaA,{2}];
CNGradLayer[SubsampleFilterBankToFilterBank,layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[SubsampleFilterBankToFilterBank,grad_]:=SubsampleFilterBankToFilterBank;
CNLayerNumberParameters[SubsampleFilterBankToFilterBank] := 0;


(*
   Layer: PadFilter
*)
SyntaxInformation[PadFilter]={"ArgumentsPattern"->{_}};
CNForwardPropogateLayer[PadFilter[padding_],inputs_]:=Map[ArrayPad[#,padding,.0]&,inputs];
CNBackPropogateLayer[PadFilter[padding_],postLayerDeltaA_,_,_]:=
   postLayerDeltaA[[All,padding+1;;-padding-1,padding+1;;-padding-1]];
CNGradLayer[PadFilter[padding_],layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[PadFilter[padding_],grad_]:=PadFilter[padding];
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
CNBackPropogateLayer[DropoutLayerMask[mask_],postLayerDeltaA_,_,_]:=mask*postLayerDeltaA;
CNGradLayer[DropoutLayerMask[mask_],layerInputs_,layerOutputDelta_]:={};
CNGradLayer[DropoutLayer[_,_],layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[networkLayer_DropoutLayer,grad_]:=DropoutLayer[networkLayer[[1]],networkLayer[[2]]];
CNLayerNumberParameters[DropoutLayer[_,_]] := 0;


(*
   Default Description
*)
CNLayerDescription[layer_Symbol]:=ToString[layer];
CNLayerDescription[layer_]:=ToString[Head[layer]];
