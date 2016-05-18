(* ::Package:: *)

SyntaxInformation[CNDreamLoss]={"ArgumentsPattern"->{_}};


CNDeltaLoss[CNDreamLoss,outputs_,targets_]:=targets*1.;


Options[Dream] = { MaxIterations -> 2000, StepSize->.01 , StepMonitor-> (#&), Momentum->0.0, MomentumType->"Classic" };


Dream[net_,inputDims_,neuron_,opts:OptionsPattern[]]:=( 
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
