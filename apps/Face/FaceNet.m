(* ::Package:: *)

<<CNNeuralCore.m


<<CNObjectLocalization.m


FaceNet = CNReadModel["FaceNet\\FaceNet2Convolve"];
GenderNet = CNReadModel["FaceNet\\GenderDetectionNet1"];


CNFaceDetection::usage = "CNFaceDetection[mirror,spaces] performs face detection.
mirror is a Bool indicating whether you wish the image to be a mirror image or not.
This setting is a personal preference. For using an external camera this would generally
be set to False. For use with an internal camera on a laptop, the setting True simulates a mirror
and is more ergonomically consistent with intuitions gained from looking into mirrors.
The spaces parameter is there to help align up the image horizontally with where the camera lies
on the laptop. This is less relevant for an external camera (set to 0).
Responsivess is strongest for full face in crop'd image, ie chin near bottom of image and top of head near top.";
CNFaceDetection[mirror_,spaces_,chartStyleF_:Function[patch,Green]] :=
   Module[{cropImage},
      CNCameraMainLoop[(
         cropImage=Map[Reverse,ImageData[#][[24-15;;24+16,32-15;;32+16]]];
         If[mirror==False,cropImage=Map[Reverse,cropImage]];
         {
            StringJoin[ConstantArray[" ",spaces]],
            BarChart[CNForwardPropogate[{cropImage},FaceNet],PlotRange->{0,1},ChartStyle->chartStyleF[cropImage]],
            Show[cropImage//Image,ImageSize->Medium]})&
      ,64]
   ]


CNFaceWithGenderDetection::usage = "CNFaceWithGenderDetection[mirror,spaces] performs same function as CNFaceDetection
but adds an attempt to determine gender which is displayed using the color of the bounding boxes.";
CNFaceWithGenderDetection[mirror_,spaces_] :=
   CNFaceDetection[mirror,spaces,Function[image,Blend[{Pink,Blue},CNForwardPropogate[{image},GenderNet][[1]]]]]


(*
   We hack FaceNet to turn it from a window based neural network into one that applies convolution across the whole of the image.
   Firstly the padding layers are removed, and the final FullyConnected layer is converted into a convolutional layer.
   This is not completely equivelent to the window based approach as the removal of padding will allow some leakage of information
   from outside the window to flow into subsequent convolutional layers. Hence the fudge factor as the first term where we are applying
   the logistic expression as the final stage. The purpose of this is to bias the net in favour of positives, and then we use the conventional
   window based approach to filter out false positives.
   Note also the second aspect (CNPriorAdjustment) where we adjust for the fact that our prior probabilities have changed. I assume there are not
   thousands of faces in a given image, therefore for any particular window the prior probability of a positive is rather low. Technically the
   1/301 factor might arguably (depending on assumptions) be expected to vary with the size of the image. In practice this has been found to work
   quite well with images sizes up to 640.
*)
CNFaceLocalizationConvolve::usage = "CNFaceLocalizationConvolve[image, colorStyleF] draws bounding boxes around faces found within the image
using FaceNet neural network. The faces are assumed to fit within a 32*32 sliding window.";
CNFaceLocalizationConvolve[image_?CNImageQ,colorStyleF_] := (
   HackedFaceNetConvolve1 = Append[
      Delete[FaceNet,{{1},{5},{9}}][[1;;9]],
      ConvolveFilterBankTo2D[0.,unflatten[FaceNet[[-2,2]],{64,4,4}]]];
   facemap = CNLogisticFn[-7.503736 + CNForwardPropogate[image,HackedFaceNetConvolve1]];
   extractFacePositions = Position[facemap,q_/;q>.5];
   originalCoordsFacePositions = Map[(({#[[2]],#[[1]]}-{1,1})*8+{14,14} + {16,16})&,extractFacePositions];
   filteredFacePositions = Select[originalCoordsFacePositions,CNPriorAdjustment[0.5,1./301,CNForwardPropogate[CNGetPatch[image,#],FaceNet]]>.5&];
   Map[CNOutlineGraphics[CNBoundingRectangles[{{#[[1]],ImageDimensions[image][[2]]-#[[2]]}},{16,16}],colorStyleF[CNGetPatch[image,#]]]&,filteredFacePositions]
);


CNFaceLocalization::usage = "CNFaceLocalization[image,colorStyleF] searches for faces within the image and draws boxes around them.
It searches at multiple scales, and the color can be changed by passing in a ColorStyle function (which receives a 32*32 window as an argument).";
CNFaceLocalization[image_?CNImageQ,colorStyleF_:Function[{patch},Green]] := (
   Show[image,
      Table[CNRescaleGraphics[CNFaceLocalizationConvolve[#,colorStyleF]&,image,.8^sc],{sc,0,-3+(Log[32]-Log[Min[ImageDimensions[image]]])/Log[.8]}]
   (* Slight hack with -3 factor above. Ideally fixup CNFaceLocalizationConvolve so it can handle 32*32 inputs *)
   ]
);


CNFaceLocalization[image_?CNImageQ,colorStyleF_:Function[{patch},Green]] :=
   CNObjectLocalization[ image, FaceNet, {Threshold->.997, ColorStyleF->colorStyleF} ]
(* Note Solve[ CNPriorAdjustment[ 0.5, 1./301,x]\[Equal]0.5,x] gives approx .997 *)


CNFaceWithGenderLocalization::usage = "CNFaceWithGenderLocalization[cnimage] searches for faces at multipe scales within the image and attempts gender recognition.";
CNFaceWithGenderLocalization[image_?CNImageQ] :=
   CNFaceLocalization[image,Function[{patch},Blend[{Pink,Blue},CNForwardPropogate[patch,GenderNet]]]]
