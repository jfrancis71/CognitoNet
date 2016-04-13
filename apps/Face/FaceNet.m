(* ::Package:: *)

<<CNNeuralCore.m


(* Read in the pretrained models and save them with different names so they
   are not overwritten.
*)
CNRead["FaceNet\\FaceDetectionNet1"]; FaceNet = CurrentModel;
CNRead["FaceNet\\GenderDetectionNet1"]; GenderNet = CurrentModel;


CNFaceDetection[mirror_,spaces_,chartStyleF_] :=
   Module[{cropImage},
      CNCameraMainLoop[(
         cropImage=Map[Reverse,#[[24-15;;24+16,32-15;;32+16]]];
         If[mirror==False,cropImage=Map[Reverse,cropImage]];
         {
            StringJoin[ConstantArray[" ",spaces]],
            BarChart[CNForwardPropogate[{cropImage},FaceNet],PlotRange->{0,1},ChartStyle->chartStyleF[cropImage]],
            cropImage//CNImage})&
      ,64]
   ]


CNFaceDetection::usage = "CNFaceDetection[mirror,spaces] performs face detection.
mirror is a Bool indicating whether you wish the image to be a mirror image or not.
This setting is a personal preference. For using an external camera this would generally
be set to False. For use with an internal camera on a laptop, the setting True simulates a mirror
and is more ergonomically consistent with intuitions gained from looking into mirrors.
The spaces parameter is there to help align up the image horizontally with where the camera lies
on the laptop. This is less relevant for an external camera (set to 0).
Responsivess is great for full face in crop'd image, ie chin near bottom of image and top of head near top.";
CNFaceDetection[mirror_,spaces_] :=
   CNFaceDetection[mirror,spaces,Function[image,Green]];


CNFaceAndGenderDetection::usage = "CNFaceDetection[mirror,spaces] performs face detection.
mirror is a Bool indicating whether you wish the image to be a mirror image or not.
This setting is a personal preference. For using an external camera this would generally
be set to False. For use with an internal camera on a laptop, the setting True simulates a mirror
and is more ergonomically consistent with intuitions gained from looking into mirrors.
The spaces parameter is there to help align up the image horizontally with where the camera lies
on the laptop. This is less relevant for an external camera (set to 0).
Responsivess is great for full face in crop'd image, ie chin near bottom of image and top of head near top.";
CNFaceAndGenderDetection[mirror_,spaces_] :=
   CNFaceDetection[mirror,spaces,Function[image,Blend[{Pink,Blue},CNForwardPropogate[{image},GenderNet][[1,1]]]]]
