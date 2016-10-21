(* ::Package:: *)

(*
   Layer: FullyConnected1DToScalar
*)
SyntaxInformation[FullyConnected1DToScalar]={"ArgumentsPattern"->{_,_}};
(* For below justification, see http://arxiv.org/pdf/1206.5533v2.pdf page 15
   Yoshua Bengio Practical Recommendations for Gradient-Based Training of Deep Architectures
*)
FullyConnected1DToScalarInit[noFromNeurons_]:=
   FullyConnected1DToScalar[.0,Table[Random[]-.5,{noFromNeurons}]/Sqrt[noFromNeurons]]
CNForwardPropogateLayer[FullyConnected1DToScalar[layerBias_,layerWeights_?VectorQ],
inputs_List] := (
   inputs.layerWeights + layerBias
   (* Note rows in inputs correspond to examples,
      and cols to neurons. + operator adds layerBiases columnwise to the matrix, hence
      these transposes. *)
);
CNBackPropogateLayer[FullyConnected1DToScalar[bias_,weights_],postLayerDeltaA_,_,_] :=
   Transpose[{postLayerDeltaA}].{weights};
CNGradLayer[FullyConnected1DToScalar[bias_,weights_],layerInputs_,layerOutputDelta_] :=
   {Total[layerOutputDelta],Transpose[layerInputs].layerOutputDelta};
CNLayerWeightPlus[networkLayer_FullyConnected1DToScalar,grad_] :=
   FullyConnected1DToScalar[networkLayer[[1]]+grad[[1]],networkLayer[[2]]+grad[[2]]];
CNLayerNumberParameters[FullyConnected1DToScalar[layerBias_,layerWeights_?VectorQ]] :=
   Length[layerWeights] + 1;


(*
   Layer: FullyConnected1DTo1D
*)
SyntaxInformation[FullyConnected1DTo1D]={"ArgumentsPattern"->{_,_}};
FullyConnected1DTo1DInit[noFromNeurons_,noToNeurones_] :=
   FullyConnected1DTo1D[ConstantArray[0.,noToNeurones],
      Table[Random[]-.5,{noToNeurones},{noFromNeurons}]/Sqrt[noFromNeurons]];
CNForwardPropogateLayer[FullyConnected1DTo1D[layerBiases_List,layerWeights_?MatrixQ],
inputs_List] := (
   CNAssertAbort[(layerWeights//First//Length)==(Transpose[inputs]//Length),
      "FullyConnected1DTo1D::Weight-Activation Error. Input length inconsistent with
 weight matrix."];
   CNAssertAbort[(layerBiases//Length)==(layerWeights//Length),
      "FullyConnected1DTo1D::Weight-Weight Error. Layer specification internally
 inconsistent."];
   Transpose[layerWeights.Transpose[inputs] + layerBiases] (* Note rows in inputs
correspond to examples, and cols to neurons. + operator adds layerBiases columnwise to the
matrix, hence these transposes. *)
);
CNBackPropogateLayer[FullyConnected1DTo1D[biases_,weights_],postLayerDeltaA_,_,_] :=
   postLayerDeltaA.weights
CNGradLayer[FullyConnected1DTo1D[biases_,weights_],layerInputs_,layerOutputDelta_] :=
   {Total[Transpose[layerOutputDelta],{2}],Transpose[layerOutputDelta].layerInputs};
CNLayerWeightPlus[networkLayer_FullyConnected1DTo1D,grad_] :=
   FullyConnected1DTo1D[networkLayer[[1]]+grad[[1]],networkLayer[[2]]+grad[[2]]];
CNLayerNumberParameters[FullyConnected1DTo1D[layerBiases_List,layerWeights_?MatrixQ]] :=
   Length[Flatten[layerWeights]] + Length[layerBiases];


(*
   Layer: Convolve2D
*)
SyntaxInformation[Convolve2D]={"ArgumentsPattern"->{_,_}};
CNForwardPropogateLayer[Convolve2D[layerBias_,layerKernel_],inputs_] := (
   ListCorrelate[{layerKernel},inputs]+layerBias
);
CNBackPropogateLayer[Convolve2D[biases_,weights_],postLayerDeltaA_,_,_] :=
   Table[ListConvolve[weights,postLayerDeltaA[[t]],{+1,-1},0],
      {t,1,Length[postLayerDeltaA]}];
CNGradLayer[Convolve2D[biases_,weights_],layerInputs_,layerOutputDelta_] :=
   {Total[layerOutputDelta,3],
   Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta,layerInputs}]]};
CNLayerWeightPlus[networkLayer_Convolve2D,grad_] :=
   Convolve2D[networkLayer[[1]]+grad[[1]],networkLayer[[2]]+grad[[2]]];
CNLayerNumberParameters[Convolve2D[layerBias_,layerKernel_]] :=
   Length[Flatten[layerKernel]] + 1;


(*
   Layer: Convolve2DToFilterBank
*)
SyntaxInformation[Convolve2DToFilterBank]={"ArgumentsPattern"->{_}};
CNForwardPropogateLayer[Convolve2DToFilterBank[filters_],inputs_] := (
   CNAssertAbort[(inputs[[1]]//Dimensions//Length)==2,
      "Convolve2DToFilterBank::inputs does not match 2D structure"];
   Transpose[Map[CNForwardPropogateLayer[#,inputs]&,filters],{2,1,3,4}]
);
Convolve2DToFilterBankInit[noNewFilterBank_,filterSize_]:=
   Convolve2DToFilterBank[
      Table[Convolve2D[0.,
            Table[Random[]-.5,{filterSize},{filterSize}]/Sqrt[filterSize*filterSize]],
         {noNewFilterBank}]]
CNBackPropogateLayer[Convolve2DToFilterBank[filters_],postLayerDeltaA_,inputs_,outputs_] :=
   Sum[CNBackPropogateLayer[filters[[f]],postLayerDeltaA[[All,f]],inputs,outputs],
      {f,1,Length[filters]}]
CNGradLayer[Convolve2DToFilterBank[filters_],layerInputs_,layerOutputDelta_] :=
   Table[{
      Total[layerOutputDelta[[All,filterIndex]],3],
      Apply[Plus,MapThread[
         ListCorrelate,
         {layerOutputDelta[[All,filterIndex]],layerInputs}]]},
      {filterIndex,1,Length[filters]}];
CNLayerWeightPlus[networkLayer_Convolve2DToFilterBank,grad_] :=
   Convolve2DToFilterBank[CNLayerWeightPlus[networkLayer[[1]],grad]];
CNLayerNumberParameters[Convolve2DToFilterBank[filters_]] :=
   Total[Map[CNLayerNumberParameters,filters]];


(*
   Layer: ConvolveFilterBankTo2D
*)
SyntaxInformation[ConvolveFilterBankTo2D]={"ArgumentsPattern"->{_,_}};
CNForwardPropogateLayer[ConvolveFilterBankTo2D[bias_,kernels_],inputs_] := (
   CNAssertAbort[Length[inputs[[1]]]==Length[kernels],
      "ConvolveFilterBankTo2D::#Kernels ("<>ToString[Length[kernels]]<>") not equal
to #Features ("<>ToString[Length[inputs[[1]]]]<>") in input feature map"];
   bias+Sum[ListCorrelate[{kernels[[kernel]]},inputs[[All,kernel]]],
      {kernel,1,Length[kernels]}]);
ConvolveFilterBankTo2DInit[noOldFilterBank_,filterSize_] :=
      ConvolveFilterBankTo2D[0.,
            Table[Random[]-.5,{noOldFilterBank},{filterSize},{filterSize}]/
               Sqrt[noOldFilterBank*filterSize*filterSize]]
CNBackPropogateLayer[ConvolveFilterBankTo2D[bias_,kernels_],postLayerDeltaA_,_,_] := (
   Transpose[Table[ListConvolve[{kernels[[w]]},postLayerDeltaA,{+1,-1},0],
      {w,1,Length[kernels]}],{2,1,3,4}]);
CNGradLayer[ConvolveFilterBankTo2D[bias_,kernels_],layerInputs_,layerOutputDelta_] := (
   (*{Total[layerOutputDelta,3],Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta,layerInputs}]]}*)
   {Total[layerOutputDelta,3],
   Table[Apply[Plus,MapThread[ListCorrelate,{layerOutputDelta,layerInputs[[All,w]]}]],
   {w,1,Length[kernels]}]});
CNLayerWeightPlus[networkLayer_ConvolveFilterBankTo2D,grad_] :=
   ConvolveFilterBankTo2D[networkLayer[[1]]+grad[[1]],networkLayer[[2]]+grad[[2]]];
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
CNForwardPropogateLayer[ConvolveFilterBankToFilterBank[filters_],inputs_] :=
   Module[{i1,i2,i3},
      i1 = Map[Partition[#,Dimensions[filters[[1,2,1]]],{1,1}]&,inputs,{2}];
      i2=(Map[Flatten,
            Transpose[i1,{1,4,2,3,5,6}],{3}].
         Transpose[Map[Flatten,filters[[All,2]]]]);
      i3=Transpose[i2,{1,3,4,2}];
      Do[i3[[All,t]]=i3[[All,t]]+filters[[All,1]][[t]],{t,1,Length[i3[[1]]]}];i3
];
ConvolveFilterBankToFilterBankInit[noOldFilterBank_,noNewFilterBank_,filterSize_] :=
   ConvolveFilterBankToFilterBank[
      Table[ConvolveFilterBankTo2D[0.,
            Table[Random[]-.5,{noOldFilterBank},{filterSize},{filterSize}]/
               Sqrt[noOldFilterBank*filterSize*filterSize]],
            {noNewFilterBank}]]
CNBackPropogateLayer[ConvolveFilterBankToFilterBank[filters_],postLayerDeltaA_,inputs_,
outputs_] :=
   Sum[CNBackPropogateLayer[filters[[f]],postLayerDeltaA[[All,f]],inputs,outputs],
   {f,1,Length[filters]}];
CNGradLayer[ConvolveFilterBankToFilterBank[filters_],layerInputs_,layerOutputDelta_] :=
   Table[{
      Total[layerOutputDelta[[All,filterOutputIndex]],3],
      ListCorrelate[Transpose[{layerOutputDelta[[All,filterOutputIndex]]},{2,1,3,4}],
         layerInputs][[1]]},
      {filterOutputIndex,1,Length[filters]}]
CNLayerWeightPlus[ConvolveFilterBankToFilterBank[filters_],grad_] :=
   ConvolveFilterBankToFilterBank[CNLayerWeightPlus[filters,grad]];
CNLayerNumberParameters[ConvolveFilterBankToFilterBank[filters_]] :=
   Total[Map[CNLayerNumberParameters,filters]];


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
   postLayerDeltaA*Boole[Positive[inputs]];
CNGradLayer[ReLU,layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[ReLU,grad_]:=ReLU;


(*
   Layer: MaxPoolingFilterBankToFilterBank
*)
SyntaxInformation[MaxPoolingFilterBankToFilterBank]={"ArgumentsPattern"->{}};
CNForwardPropogateLayer[MaxPoolingFilterBankToFilterBank,inputs_] :=
   Map[Function[image,Map[Max,Partition[image,{2,2}],{2}]],inputs,{2}];
UpSample[x_]:=Riffle[temp=Riffle[x,x]//Transpose;temp,temp]//Transpose;
backRoutingSingleMax[ backRoute_ ] := (* Takes a 2D slice of back routes and makes sure there is only one max back (like TensorFlow) *)
   Boole[Positive[ArrayFlatten[
      Map[(
         {{#[[1,1]],#[[1,2]]-#[[1,1]]},{#[[2,1]]-(#[[1,1]]+#[[1,2]]),#[[2,2]]-(#[[1,1]]+#[[1,2]]+#[[2,1]])}}
         )&,Partition[backRoute,{2,2}],{2}]]]];
backRouting[previousZ_,nextA_] := 
   Map[backRoutingSingleMax,UnitStep[previousZ-Map[UpSample,nextA,{2}]],{2}];
CNBackPropogateLayer[MaxPoolingFilterBankToFilterBank,postLayerDeltaA_,layerInputs_,
layerOutputs_]:=
   ( backRouting[layerInputs,layerOutputs]*Map[UpSample,postLayerDeltaA,{2}] )
CNGradLayer[MaxPoolingFilterBankToFilterBank,layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[MaxPoolingFilterBankToFilterBank,grad_] :=
   MaxPoolingFilterBankToFilterBank;
CNLayerNumberParameters[MaxPoolingFilterBankToFilterBank] := 0;


Needs["Developer`"]
(*
   Layer: MaxConvolveFilterBankToFilterBank
*)
SyntaxInformation[MaxConvolveFilterBankToFilterBank]={"ArgumentsPattern"->{}};
CNForwardPropogateLayer[MaxConvolveFilterBankToFilterBank,inputs_]:=
   Map[Max,Map[Partition[#,{3,3},{1,1},{-2,+2},-2.0]&,inputs,{2}],{4}];
CNBackPropogateLayer[MaxConvolveFilterBankToFilterBank,postLayerDeltaA_,inputs_,outputs_] :=
   (
   CNAssertAbort[Max[inputs]<1.4,"BackPropogateLayer::
MaxConvolveFilterBankToFilterBank algo not designed for inputs > 1.4"];
(*   u1=Map[Partition[#,{3,3},{1,1},{-2,+2},-2.0]&,inputs,{2}];
   u2=Map[Max[Flatten[#]]&,u1,{4}];*)
   CNTimer["MaxConvolveFilterBankToFilterBank::u3",
      u3=ToPackedArray[Map[Partition[#,{3,3},{1,1},{-2,+2},1.5]&,outputs,{2}]]];
   CNTimer["MaxConvolveFilterBankToFilterBank::u4",
      u4=UnitStep[inputs-u3]];
   CNTimer["MaxConvolveFilterBankToFilterBank::u5",
      u5=ToPackedArray[Map[Partition[#,{3,3},{1,1},{-2,+2},-2.0]&,postLayerDeltaA,{2}]]];
   CNTimer["MaxConvolveFilterBankToFilterBank::u6",u6=u4*u5];
   CNTimer["MaxConvolveFilterBankToFilterBank::u7",u7=Map[Total[Flatten[#]]&,u6,{4}]])
CNGradLayer[MaxConvolveFilterBankToFilterBank,layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[MaxConvolveFilterBankToFilterBank,grad_ ]:=
   MaxConvolveFilterBankToFilterBank;
CNLayerNumberParameters[MaxConvolveFilterBankToFilterBank] := 0;


(*
   Layer: Softmax
*)
SyntaxInformation[Softmax]={"ArgumentsPattern"->{}};
CNForwardPropogateLayer[Softmax,inputs_] := Map[Exp[#]/Total[Exp[#]]&,inputs];
CNBackPropogateLayer[Softmax,postLayerDeltaA_,_,outputs_]:=
   Table[
      Sum[postLayerDeltaA[[n,i]]*outputs[[n,i]]*(KroneckerDelta[j,i]-outputs[[n,j]]),
         {i,1,Length[postLayerDeltaA[[1]]]}],
         {n,1,Length[postLayerDeltaA]},
      {j,1,Length[postLayerDeltaA[[1]]]}];
CNGradLayer[Softmax,layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[Softmax,grad_]:=Softmax;
CNLayerNumberParameters[Softmax] := 0;


SyntaxInformation[Adaptor2DTo1D]={"ArgumentsPattern"->{_}};
CNForwardPropogateLayer[Adaptor2DTo1D[width_],inputs_]:=(
   CNAssertAbort[(inputs[[1,1]]//Length)==width,"Adaptor2DTo1D::widths of inputs does not
match Adaptor width"];
   Map[Flatten,inputs]
);
CNBackPropogateLayer[Adaptor2DTo1D[width_],postLayerDeltaA_,_,_]:=
   Map[Partition[#,width]&,postLayerDeltaA];
CNGradLayer[Adaptor2DTo1D[width_],layerInputs_,layerOutputDelta_]:={}
CNLayerWeightPlus[networkLayer_Adaptor2DTo1D,grad_]:=Adaptor2DTo1D[networkLayer[[1]]];
CNLayerNumberParameters[Adaptor2DTo1D[_]] := 0;


(*
   Layer: AdaptorFilterBankTo1D
*)
(* Helper Function sources from Mathematica on-line documentation regarding example use of
Partition *)
unflatten[e_,{d__?((IntegerQ[#]&&Positive[#])&)}]:= 
   Fold[Partition,e,Take[{d},{-1,2,-1}]] /;(Length[e]===Times[d]);
SyntaxInformation[AdaptorFilterBankTo1D]={"ArgumentsPattern"->{_,_,_}};
CNForwardPropogateLayer[AdaptorFilterBankTo1D[features_,width_,height_],inputs_]:=(
   Map[Flatten,inputs]
);
CNBackPropogateLayer[AdaptorFilterBankTo1D[features_,width_,height_],
postLayerDeltaA_,_,_] :=
   unflatten[Flatten[postLayerDeltaA],{Length[postLayerDeltaA],features,width,height}];
CNGradLayer[AdaptorFilterBankTo1D[features_,width_,height_],
   layerInputs_,layerOutputDelta_] := {};
CNLayerWeightPlus[networkLayer_AdaptorFilterBankTo1D,grad_] :=
   AdaptorFilterBankTo1D[networkLayer[[1]],networkLayer[[2]],networkLayer[[3]]];
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
CNForwardPropogateLayer[PadFilterBank[padding_],inputs_] :=
   Map[ArrayPad[#,padding,.0]&,inputs,{2}];
CNBackPropogateLayer[PadFilterBank[padding_],postLayerDeltaA_,_,_]:=
   postLayerDeltaA[[All,All,padding+1;;-padding-1,padding+1;;-padding-1]];
CNGradLayer[PadFilterBank[padding_],layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[PadFilterBank[padding_],grad_]:=PadFilterBank[padding];
CNLayerNumberParameters[PadFilterBank[padding_]] := 0;


(*
   Layer: LogSumExp
   Bit like a smooth approximation of a max function
   Ref: https://en.wikipedia.org/wiki/LogSumExp

   Ref: http://arxiv.org/pdf/1411.6228v3.pdf
   From Image-level to Pixel-level Labelling with Convolutional Networks
   Pinheiro and Collobert

   My motivational use was for the NoEntry traffic detector where there is positional uncertainty.
   I model as either traffic sign not present or probability as present is sum of probabilities over all
   locations where it could be present. If you take the softmax function and then model over this sum
   you end up with a LogSumExp function (wiht probabilities in the log domain).
*)
SyntaxInformation[LogSumExp] = {"ArgumentsPattern"->{_}};
CNForwardPropogateLayer[LogSumExp,inputs_] :=
   Map[Log[Total[Exp[#]]]&,inputs];
CNBackPropogateLayer[LogSumExp,postLayerDeltaA_,inputs_,outputs_]:=
   Table[postLayerDeltaA[[ex]]*Exp[inputs[[ex]]]/Total[Exp[inputs[[ex]]]],{ex,1,Length[postLayerDeltaA]}];
CNGradLayer[LogSumExp,layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[LogSumExp,grad_]:=LogSumExp;
CNLayerNumberParameters[LogSumExp] := 0;


(*
   Layer: Max1D
*)
SyntaxInformation[Max1D] = {"ArgumentsPattern"->{_}};
CNForwardPropogateLayer[Max1D,inputs_] :=
   Map[Max,inputs];
CNBackPropogateLayer[Max1D,postLayerDeltaA_,inputs_,outputs_]:=
   Table[postLayerDeltaA[[ex]]*UnitStep[outputs[[ex]]-inputs[[ex]]],{ex,1,Length[postLayerDeltaA]}];
CNGradLayer[Max1D,layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[Max1D,grad_]:=Max1D;
CNLayerNumberParameters[Max1D] := 0;


(*
   Layer: Mean1D
*)
SyntaxInformation[Mean1D] = {"ArgumentsPattern"->{_}};
CNForwardPropogateLayer[Mean1D,inputs_] :=
   Map[Mean,inputs];
CNBackPropogateLayer[Mean1D,postLayerDeltaA_,inputs_,outputs_]:=
   Table[postLayerDeltaA[[ex]]*(1/Length[inputs[[ex]]]),{ex,1,Length[postLayerDeltaA]},{n,1,Length[inputs[[ex]]]}];
CNGradLayer[Mean1D,layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[Mean1D,grad_]:=Mean1D;
CNLayerNumberParameters[Mean1D] := 0;


(*
   Layer: SubsampleFilterBankToFilterBank
*)
SyntaxInformation[SubsampleFilterBankToFilterBank] = {"ArgumentsPattern"->{}};
CNForwardPropogateLayer[SubsampleFilterBankToFilterBank,inputs_] :=
   Map[#[[1;;-1;;2,1;;-1;;2]]&,inputs,{2}];
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
SyntaxInformation[CNDropoutLayer]={"ArgumentsPattern"->{_,_}};
SyntaxInformation[DropoutLayerMask]={"ArgumentsPattern"->{_}};
Dropout[layer_,inputs_]:=layer;
Dropout[net_List,inputs_]:=Map[Dropout[#,inputs]&,net];
Dropout[CNDropoutLayer[dims_,dropoutProb_],inputs_]:=(
   DropoutLayerMask[Array[(RandomInteger[])&,Prepend[dims,Length[inputs]]]]);
CNForwardPropogateLayer[CNDropoutLayer[_,_],inputs_]:=0.5*inputs;
CNForwardPropogateLayer[DropoutLayerMask[mask_],inputs_]:=inputs*mask;
CNBackPropogateLayer[DropoutLayerMask[mask_],postLayerDeltaA_,_,_]:=mask*postLayerDeltaA;
CNBackPropogateLayer[CNDropoutLayer[_,_],postLayerDeltaA_,_,_]:=0.5*postLayerDeltaA;
CNGradLayer[DropoutLayerMask[mask_],layerInputs_,layerOutputDelta_]:={};
CNGradLayer[CNDropoutLayer[_,_],layerInputs_,layerOutputDelta_]:={};
CNLayerWeightPlus[networkLayer_CNDropoutLayer,grad_] :=
   CNDropoutLayer[networkLayer[[1]],networkLayer[[2]]];
CNLayerNumberParameters[CNDropoutLayer[_,_]] := 0;


CNForwardPropogateLayer[DecorrelationRegularizer, inputs_List] :=  inputs;
CNBackPropogateLayer[DecorrelationRegularizer,postLayerDeltaA_,layerInputs_,_] := (
   If[Length[layerInputs]==1,postLayerDeltaA,
      CNCovar= Covariance[layerInputs];
      CNMn = Mean[layerInputs];
      CNI=Map[(#-CNMn)&,layerInputs];
      postLayerDeltaA + (bt=1*(1/Length[postLayerDeltaA])*Table[CNCovar[[a]].ReplacePart[CNI[[m]],a->0.],{m,1,Length[postLayerDeltaA]},{a,1,4096}])] )
CNGradLayer[DecorrelationRegularizer,layerInputs_,layerOutputDelta_] :=
   {};
CNLayerWeightPlus[DecorrelationRegularizer,grad_] :=
    DecorrelationRegularizer ;
CNLayerNumberParameters[DecorrelationRegularizer] := 0


(*
   Default Description
*)
CNLayerDescription[layer_Symbol]:=ToString[layer];
CNLayerDescription[layer_]:=ToString[Head[layer]];


CNRegularizeLayer[ Convolve2D[ _, _ ], { bias_, kernel_ }, opts:OptionsPattern[] ] := { bias, kernel + 2.*kernel*OptionValue[ L2W ] + Sign[kernel] *OptionValue[L1W] };
CNRegularizeLayer[ Convolve2DToFilterBank[ filters_ ], grads_, opts:OptionsPattern[] ] := MapThread[ CNRegularizeLayer[ #1, #2, opts  ]&, { filters, grads }];
CNRegularizeLayer[ ConvolveFilterBankTo2D[ _,_ ], { bias_, kernel_ }, opts:OptionsPattern[] ] := ( { bias, kernel + 2.*kernel*OptionValue[ L2W ]  + Sign[kernel] *OptionValue[L1W] });
CNRegularizeLayer[ ConvolveFilterBankToFilterBank[ filters_ ], grads_, opts:OptionsPattern[] ] := ( MapThread[ CNRegularizeLayer[ #1, #2, opts  ]&, { filters, grads } ]);

CNRegularizeLayer[ GPUConvolve2D[ _, _ ], { bias_, kernel_ }, opts:OptionsPattern[] ] := { bias, kernel + 2.*kernel*OptionValue[ L2W ]  + Sign[kernel] *OptionValue[L1W]};
CNRegularizeLayer[ GPUConvolve2DToFilterBank[ filters_ ], grads_, opts:OptionsPattern[] ] := ( MapThread[ CNRegularizeLayer[ #1, #2, opts  ]&, { filters, grads } ] )
CNRegularizeLayer[ GPUConvolveFilterBankTo2D[ _,_ ], { bias_, kernel_ }, opts:OptionsPattern[] ] := ({ bias, kernel + 2.*kernel*OptionValue[ L2W ]  + Sign[kernel] *OptionValue[L1W] });
CNRegularizeLayer[ GPUConvolveFilterBankToFilterBank[ filters_ ], grads_, opts:OptionsPattern[] ] := (MapThread[ CNRegularizeLayer[ #1, #2, opts  ]&, { filters, grads } ]);


CNRegularizeLayer[ netLayer_, gradsLayer_, opts:OptionsPattern[] ] := ( a1 = netLayer; a2 = gradsLayer; a3 = opts; gradsLayer )
