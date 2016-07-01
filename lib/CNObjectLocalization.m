(* ::Package:: *)

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


CNObjectLocalizationConvolve[image_,net_] := ( 
   map=CNPriorAdjustment[0.5,1/300., CNForwardPropogate[image,net]];
extractPositions=Position[map,x_/;x>.5];
origCoords=Map[({#[[2]],#[[1]]}-{1,1})*4&,extractPositions];
Map[CNOutlineGraphics[CNBoundingRectangles[{{16,-16}+{#[[1]],ImageDimensions[image][[2]]-#[[2]]}},{16,16}]]&,origCoords]
)


CNObjectLocalization[image_?CNImageQ,net_] := ( 
   Show[image,
      Table[CNRescaleGraphics[CNObjectLocalizationConvolve[#,net]&,image,.8^sc],{sc,0,-3+(Log[32]-Log[Min[ImageDimensions[image]]])/Log[.8]}]
   (* Slight hack with -3 factor above. Ideally fixup CNFaceLocalizationConvolve so it can handle 32*32 inputs *)
   ]
);


(*
Note we are implicitly assuming that the final layers were an Adaptor2DTo1D followed by a LogSum and then a logistic
*)
CNConvertLogSumModelToConvolve[model_]:=Join[
model[[1;;-4]],
{Logistic}];
