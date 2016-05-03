(* ::Package:: *)

<<CNNeuralCore.m


FaceNet = CNReadModel["FaceNet\\FaceDetectionNet1"];
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
         cropImage=Map[Reverse,ImageData[#,DataReversed->True][[24-15;;24+16,32-15;;32+16]]];
         If[mirror==False,cropImage=Map[Reverse,cropImage]];
         {
            StringJoin[ConstantArray[" ",spaces]],
            BarChart[CNForwardPropogate[{cropImage},FaceNet],PlotRange->{0,1},ChartStyle->chartStyleF[cropImage]],
            cropImage//Raster//Graphics})&
      ,64]
   ]


CNFaceWithGenderDetection::usage = "CNFaceWithGenderDetection[mirror,spaces] performs same function as CNFaceDetection
but adds an attempt to determine gender which is displayed using the color of the bounding boxes.";
CNFaceWithGenderDetection[mirror_,spaces_] :=
   CNFaceDetection[mirror,spaces,Function[image,Blend[{Pink,Blue},CNForwardPropogate[{image},GenderNet][[1,1]]]]]


CNGetPatch[image_,coords_] := Image[ImageData[image,DataReversed->True][[coords[[2]]-16;;coords[[2]]+15,coords[[1]]-16;;coords[[1]]+15]]//Reverse]


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
      ConvolveFilterBankTo2D[0.,unflatten[FaceNet[[-2,2,1]],{64,4,4}]]];
   facemap = CNLogisticFn[-7.503736 + CNForwardPropogate[image,HackedFaceNetConvolve1]];
   extractFacePositions = Position[facemap,q_/;q>.5];
   originalCoordsFacePositions = Map[(({#[[2]],#[[1]]}-{1,1})*8+{14,14} + {16,16})&,extractFacePositions];
   filteredFacePositions = Select[originalCoordsFacePositions,CNPriorAdjustment[0.5,1./301,CNForwardPropogate[CNGetPatch[image,#],FaceNet][[1]]]>.5&];
   Map[CNOutlineGraphics[CNBoundingRectangles[{#},{16,16}],colorStyleF[CNGetPatch[image,#]]]&,filteredFacePositions]
);


CNFaceLocalization::usage = "CNFaceLocalization[image,colorStyleF] searches for faces within the image and draws boxes around them.
It searches at multiple scales, and the color can be changed by passing in a ColorStyle function (which receives a 32*32 window as an argument).";
CNFaceLocalization[image_?CNImageQ,colorStyleF_:Function[{patch},Green]] := (
   Show[image,
      Table[CNRescaleGraphics[CNFaceLocalizationConvolve[#,colorStyleF]&,image,.8^sc],{sc,0,-3+(Log[32]-Log[Min[ImageDimensions[image]]])/Log[.8]}]
   (* Slight hack with -3 factor above. Ideally fixup CNFaceLocalizationConvolve so it can handle 32*32 inputs *)
   ]
);


CNFaceWithGenderLocalization::usage = "CNFaceWithGenderLocalization[cnimage] searches for faces at multipe scales within the image and attempts gender recognition.";
CNFaceWithGenderLocalization[image_?CNImageQ] :=
   CNFaceLocalization[image,Function[{patch},Blend[{Pink,Blue},CNForwardPropogate[patch,GenderNet][[1]]]]]


(*
   The ratio version of Bayes Theorem is most useful here:
      PosteriorRatio = PriorRatio * LikelihoodRatio

      priorRatio = prior/(1-prior) etc

   Basic idea is that the trained neural network is a discriminative network, therefore it has
   both a prior and likelihood probabilities baked into its output. This is fine if population frequency
   between training and test is the same, but if they change it will be systematically wrong.
   If we know the neural network probability output and we knew the frequency of positives in the training set
   we can back out and calculate the likelihood ratio.
   We can then use this likelihood ratio with our estimation of the test frequency of positives to calculate a new
   conditional probability that is more appropriate to our current test set.

   Example, the face database was trained with about 50% faces and 50% distractors. But this is not appropriate
   for our localization functions where we expect only a small fraction of windows to actually contain faces.

*)
CNPriorAdjustment::usage = "CNPriorAdjustment[oldPrior,newPrior,probabilities]
Adjusts neural network output probabilities if the frequency of matches in the
training scenario is different from the test scenario. probability can be a
scalar or any numeric array representing the conditional probability to be adjusted.";
CNPriorAdjustment[trainingPrior_?NumberQ,testPrior_?NumberQ,trainingPosterior_] := Module[{},
   trainingPriorRatio = trainingPrior/(1-trainingPrior);
   trainingPosteriorRatio = trainingPosterior/(1-trainingPosterior);
   likelihoodRatio = trainingPosteriorRatio/trainingPriorRatio;
   testPriorRatio = testPrior/(1-testPrior);
   testPosteriorRatio = testPriorRatio * likelihoodRatio;
   testPosterior = testPosteriorRatio/(1+testPosteriorRatio)
];
