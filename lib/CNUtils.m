(* ::Package:: *)

CNAssertAbort::usage = "CNAssertAbort[bool,message] will abort computation if bool is True (and print message).";
CNAssertAbort[bool_,message_]:=
   If[bool==False,
      Print[message];Abort[]];


CNImage::usage = "CNImage[matrix] assumes matrix represents an image structure and displays it.
The matrix is assumed to be in graphics order, i.e. row 1 of the matrix is the bottom of the image.";
CNImage[matrix_?MatrixQ]:=Graphics[Raster[matrix]];


CNImageToCNImage::usage = "CNImageToCNImage[image,width] takes a Mathematica Image object
and returns it in CNImage format (ie Raster row ordering) of specified image width";
CNImageToCNImage[image_Image,width_Integer]:=
   ImageData[ColorConvert[ImageResize[image,width],"GrayScale"]]//Reverse


CNColImage::usage = "CNColImage[data] will display image corresponding to data.
Color inputs of neural nets are typically non interleaved and in seperate plane format, e.g. 3*32*32.";
CNColImage[image_]:=Image[image,Interleaving->False]


CNCameraMainLoop::usage = "CNCameraMainLoop[programF,width] evaluates and displays programF in an infinite loop.
programF should be a function accepting a single argument which is the camera image in CNImage format.
You should specify the width of the image that programF is expecting.";
CNCameraMainLoop[programF_,width_] := 
   Module[{grOutput},Monitor[While[True,grOutput=programF[CNImageToCNImage[CurrentImage[],width]]],grOutput]];
