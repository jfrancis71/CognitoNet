(* ::Package:: *)

CNAssertAbort[bool_,message_]:=
   If[bool==False,
      Print[message];Abort[]];


CNImage[matrix_?MatrixQ]:=Graphics[Raster[matrix]];
