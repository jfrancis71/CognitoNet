(* ::Package:: *)

CNAssertAbort::usage = "CNAssertAbort[bool,message] will abort computation if bool is True
(and print message).";
CNAssertAbort[bool_,message_]:=
   If[bool==False,
      Print[message];Abort[]];


CNTimer[message_,expr_] := Module[{timer=AbsoluteTiming[expr]},
      If[MatchQ[$timers,True],Print[message," ",timer[[1]]," secs"]];timer[[2]]];
SetAttributes[CNTimer,HoldAll];


(*
Deprecated
CNImage::usage = "CNImage[matrix] assumes matrix represents an image structure and displays it.
The matrix is assumed to be in graphics order, i.e. row 1 of the matrix is the bottom of the image.";
CNImage[matrix_?MatrixQ]:=Graphics[Raster[matrix]];
*)


CNImageQ[image_] := ImageQ[image]&&ImageChannels[image]==1;


CNColImageQ[image_] := ImageQ[image]&&ImageChannels[image]==3;


CNImageListQ[images_List] := ImageQ[images[[1]]]&&ImageChannels[images[[1]]]==1


CNColImageListQ[images_List] := ImageQ[images[[1]]]&&ImageChannels[images[[1]]]==3


CNImportImage::usage = "CNImportImage[image,width] takes a Mathematica Image object
and returns a grayscale image.
CNImportImage[image,{width,height}] takes a Mathematica Image object
and returns a grayscale image of specified image width and height.
CNImportImage[file,width] reads an image file in and returns a grayscale image of specified
image width. CNImportImage[file,{width,height}] reads an image file in and returns a grayscale
image of specified image width and height. This function provides an easy way to both convert
to grayscale and resize at the same time.
";
CNImportImage[image_Image,width_Integer]:=
   ColorConvert[ImageResize[image,width],"GrayScale"]
CNImportImage[image_Image,{width_Integer,height_Integer}]:=
   ColorConvert[ImageResize[image,{width,height}],"GrayScale"]
CNImportImage[file_String,width_Integer]:=
   CNImportImage[Import[file],width];
CNImportImage[file_String,{width_Integer,height_Integer}]:=
   CNImportImage[Import[file],{width,height}];


CNReadImagesFromDirectory[directory_String,size_:128]:=
   Map[CNImportImage[#,size]&,
   FileNames[StringJoin[directory,"\\*.jpg"]]
]


CNMovieLength[file_String] := Import[file]//Length;
CNImportMovie[file_String,width_Integer] :=
   CNImportMovie[file,width, CNMovieLength[file]];
CNImportMovie[file_String,width_Integer,frames_Integer] :=
   Flatten[Map[
      Map[Function[conv,CNImportImage[conv,width]],Import[file,{"Frames",#}]]&,
         Partition[Range[frames],10,10,1,{}]],1];


(*
Deprecated
CNColImage::usage = "CNColImage[data] will display image corresponding to data.
Color inputs of neural nets are typically non interleaved and in seperate plane format, e.g. 3*32*32.";
CNColImage[image_]:=Image[image,Interleaving->False]
*)


CNImageRescale::usage = "CNImageRescale[image_,scale_] rescales image by scale factor.";
CNImageRescale[image_Image,scale_] :=
   ImageResize[image,scale*ImageDimensions[image][[1]]]


CNRescaleGraphics::usage = "CNRescaleGraphics[f,image,scale] will rescale image and pass it
into function f which is expected to produce a list of Graphics objects and then rescale back
this output.";
CNRescaleGraphics[f_Function,image_Image,scale_?NumberQ] :=
   Graphics[Map[Scale[#[[1]],1/scale,{0,0}]&,f[CNImageRescale[image,scale]]]];


CNBoundingRectangles::usage = "CNBoundingRectangles[coords, filterSize] draws a 
rectangle of dimensions specified by filterSize {width,height} centered on each point in
coords where points are specified by (y,x) with y in raster coordinates.";
CNBoundingRectangles[coords_?MatrixQ,filterSize_?VectorQ]:=
   Map[Rectangle[#-filterSize,#+filterSize]&,coords]


CNOutlineGraphics::usage = "CNOutlineGraphics[grObjects,color] produces a Graphics object
with the grObjects made transparent except for their edges.";
CNOutlineGraphics[grObjects_,Color_:Green] := Graphics[{Opacity[0],Color,
      EdgeForm[Directive[Color,Thick]],grObjects}]


CNCameraMainLoop::usage = "CNCameraMainLoop[programF,width] evaluates and displays programF in an infinite loop.
programF should be a function accepting a single argument which is the camera image.
You should specify the width of the image that programF is expecting.";
CNCameraMainLoop[programF_,width_] := 
   Module[{grOutput},Monitor[While[True,
      grOutput=programF[CNImportImage[CurrentImage[],width]]],grOutput]];


(* http://www.cs.toronto.edu/~fritz/absps/momentum.pdf *)
(* On the importance of initialization and momentum in deep learning *)
(* Sutskever, Martens, Dahl, Hinton (2013) *)
CNStepGradientDescent[ {state_, velocity_}, gradF_, plusF_, momentumDecay_, momentumType_,
   stepSize_ ] := (

   gw=If[momentumType!="Nesterov",
      gradF[state],
      gradF[plusF[state,momentumDecay*velocity]]
   ];
   newvelocity = (momentumDecay * velocity) - (gw * stepSize);
   {plusF[state, newvelocity], newvelocity}
)


SyntaxInformation[Momentum]={"ArgumentsPattern"->{}};
SyntaxInformation[MomentumType]={"ArgumentsPattern"->{}};
SyntaxInformation[StepSize]={"ArgumentsPattern"->{}};
Options[CNGradientDescent]={
   Momentum->.0,
   MomentumType->"CM",
   StepSize->.01,
   StepMonitor:>(#&)};
CNGradientDescent::usage = "NGradientDescent[state_, gradF_, plusF_, iterations_, opts]
performs gradient descent.";
CNGradientDescent[state_,gradF_,plusF_,iterations_,opts:OptionsPattern[]] :=
   First[Nest[(
      updateStep=CNStepGradientDescent[#,gradF,plusF,OptionValue[Momentum],
         OptionValue[MomentumType],OptionValue[StepSize]];
      OptionValue[StepMonitor][First[updateStep]];updateStep)&,
      {state,gradF[state]*0.0}, (* needed as if velocity has complex structure it will be
                                   otherwise initialized incorrectly *)
      iterations]]
