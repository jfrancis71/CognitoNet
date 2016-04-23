(* ::Package:: *)

CNAssertAbort::usage = "CNAssertAbort[bool,message] will abort computation if bool is True (and print message).";
CNAssertAbort[bool_,message_]:=
   If[bool==False,
      Print[message];Abort[]];


(*
Deprecated
CNImage::usage = "CNImage[matrix] assumes matrix represents an image structure and displays it.
The matrix is assumed to be in graphics order, i.e. row 1 of the matrix is the bottom of the image.";
CNImage[matrix_?MatrixQ]:=Graphics[Raster[matrix]];
*)


CNImportImage::usage = "CNImport[image,width] takes a Mathematica Image object
and returns a grayscale image.
CNImportImage[image,{width,height}] takes a Mathematica Image object
and returns a grayscale image of specified image width and height.
CNImportImage[file,width] reads an image file in and returns a grayscale image of specified image width.
CNImportImage[file,{width,height}] reads an image file in and returns a grayscale image of specified image width and height.
This function provides an easy way to both convert to grayscale and resize at the same time.
";
CNImportImage[image_Image,width_Integer]:=
   ColorConvert[ImageResize[image,width],"GrayScale"]
CNImportImage[image_Image,{width_Integer,height_Integer}]:=
   ColorConvert[ImageResize[image,{width,height}],"GrayScale"]
CNImportImage[file_String,width_Integer]:=
   CNImportImage[Import[file],width];
CNImportImage[file_String,{width_Integer,height_Integer}]:=
   CNImportImage[Import[file],{width,height}];


(*
Deprecated
CNColImage::usage = "CNColImage[data] will display image corresponding to data.
Color inputs of neural nets are typically non interleaved and in seperate plane format, e.g. 3*32*32.";
CNColImage[image_]:=Image[image,Interleaving->False]
*)


(*
Deprecated
CNImageRescale::usage = "CNImageRescale[image_,scale_] rescales image by scale factor.";
CNImageRescale[image_,scale_] :=
   ImageData[ImageResize[Image[image],scale*Length[image[[1]]]]]
*)


CNRescaleGraphics::usage = "CNRescaleGraphics[f,image,scale] will rescale image and pass it into function f
which is expected to produce a list of Graphics objects and then rescale back this output.";
CNRescaleGraphics[f_Function,image_?MatrixQ,scale_?NumberQ] :=
   Graphics[Map[Scale[#[[1]],1/scale,{0,0}]&,f[CNImageRescale[image,scale]]]];


CNBoundingRectangles::usage = "CNBoundingRectangles[coords, filterSize] draws a 
rectangle of dimensions specified by filterSize {width,height} centered on each point in coords where
points are specified by (y,x) with y in raster coordinates.";
CNBoundingRectangles[coords_?MatrixQ,filterSize_?VectorQ]:=
   Map[Rectangle[#-filterSize,#+filterSize]&,coords]


CNOutlineGraphics::usage = "CNOutlineGraphics[grObjects,color] produces a Graphics object with the
grObjects made transparent except for their edges.";
CNOutlineGraphics[grObjects_,Color_:Green]:=Graphics[{Opacity[0],Color,EdgeForm[Directive[Color,Thick]],grObjects}]


CNCameraMainLoop::usage = "CNCameraMainLoop[programF,width] evaluates and displays programF in an infinite loop.
programF should be a function accepting a single argument which is the camera image.
You should specify the width of the image that programF is expecting.";
CNCameraMainLoop[programF_,width_] := 
   Module[{grOutput},Monitor[While[True,grOutput=programF[CNImportImage[CurrentImage[],width]]],grOutput]];
