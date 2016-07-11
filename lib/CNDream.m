(* ::Package:: *)

SyntaxInformation[CNDreamLoss]={"ArgumentsPattern"->{_}};


CNDeltaLoss[CNDreamLoss,outputs_,targets_]:=targets*1.;


Options[CNDream] = { MaxIterations -> 2000, StepSize->.01 , StepMonitor-> (#&), Momentum->0.0, MomentumType->"Classic" };


CNDream[net_,inputDims_,neuron_,opts:OptionsPattern[]]:=( 
   dream=Array[.5&,inputDims];
   neuronLayer=neuron[[1]];
   target=If[Rest[neuron]!={},
      ReplacePart[
         CNForwardPropogate[{dream},net[[1;;neuronLayer]]][[1]]*.0,
         Rest[neuron]->1.0],
      1.0];
    CNGradientDescent[dream,
   (* gradient function *)
   ( 
      L=CNForwardPropogateLayers[ {#},net[[1;;neuronLayer]]];
      deltas = CNBackPropogateLayers[net[[1;;neuronLayer]],L,-CNDeltaLoss[CNDreamLoss,L[[-1]],{target}]];
      dw = CNBackPropogateLayer[net[[1]],deltas[[1]],_,_];First[dw])&
   ,Plus,OptionValue[MaxIterations],FilterRules[opts,Except[MaxIterations]]]
)


CNSalient[f_,image_?CNImageQ,model_]:=(
   L=CNForwardPropogateLayers[{ImageData[image]},model[[1;;-1]]];
   deltan=CNBackPropogateLayers[model[[1;;-1]],L,CNDeltaLoss[CNDreamLoss,L[[-1]],{f}]];
   dw=CNBackPropogateLayer[model[[1]],deltan[[1]],_,_][[1]];
   Abs[dw]/Max[Abs[dw]]
)
