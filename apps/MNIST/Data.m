(* ::Package:: *)

(*
   Loads data for the famous MNIST dataset.
   Ref: http://yann.lecun.com/exdb/mnist/
*)


<<CNUtils.m


(* Images are returned as a matrix, but in raster format. i.e. first row is bottom of image
   Display using CNImage.
*)
ReadImagesFromMNISTFile[file_String] := Module[{
   RawImageFileData,RawImageStream,RawImages},

   RawImageFileData=BinaryReadList[file,"Byte",200000000];
   RawImageStream=RawImageFileData[[17;;-1]];
   RawImages=Partition[RawImageStream,28*28];

   Map[Reverse,Map[If[#1 > 128, 1., 0.] &,Map[Partition[#,28]&,RawImages],{3}]]
];

ReadMINSTLabelFile[file_String] := Module[{
   RawClassificationFileData},

   RawClassificationFileData=BinaryReadList[file,"Byte",200000000];

   RawClassificationFileData[[9;;-1]]
];

TrainingImages = ReadImagesFromMNISTFile[$CNDataDir<>"\\MNIST\\train-images-idx3-ubyte"][[1;;59500]];
ValidationImages = ReadImagesFromMNISTFile[$CNDataDir<>"\\MNIST\\train-images-idx3-ubyte"][[59501;;60000]];
TestImages = ReadImagesFromMNISTFile[$CNDataDir<>"\\MNIST\\t10k-images-idx3-ubyte"];

TrainingLabels = ReadMINSTLabelFile[$CNDataDir<>"\\MNIST\\train-labels-idx1-ubyte"][[1;;59500]];
ValidationLabels = ReadMINSTLabelFile[$CNDataDir<>"\\MNIST\\train-labels-idx1-ubyte"][[59501;;60000]];
TestLabels = ReadMINSTLabelFile[$CNDataDir<>"\\MNIST\\t10k-labels-idx1-ubyte"];

(* Converting to the 1 of K target format *)
TrainingTargets = Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels];
ValidationTargets = Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,ValidationLabels];
TestTargets = Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TestLabels];
