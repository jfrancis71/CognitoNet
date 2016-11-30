(* ::Package:: *)

<<CNNeuralCore.m


Set::write


model=Import["C:\\Users\\julian\\Google Drive\\Personal\\Computer Science\\WebMonitor\\sockets\\History1\\SocketNet2016-09-17 17.wdx"];


<<CNGPULayers.m


CNGPUInitialize[]


<<CNObjectLocalization.m


source=Join[{1,1,1,1,1},ConstantArray[0,495]];


codebook=Table[RandomSample[source],{y,1,53},{x,1,73}];


MyMaxDetect[image_]:=If[Total[image//ImageData,2]==0,0*(image//ImageData),MaxDetect[image]//ImageData]


Demo:=CNCameraMainLoop[
(tp=#;CNObjectLocalizationConvolve[CNImportImage[#,320],CNConvertLogSumModelToConvolve[model],.997];thinned=MyMaxDetect[Threshold[Blur[map//Image],.8]];
code=UnitStep[codebook[[1,1]]*0+(Extract[codebook,Position[thinned,1]]//Total)-.01];
decode=UnitStep[Table[codebook[[y,x]].code,{y,1,53},{x,1,73}]-4.99];
{#,thinned//Image,Partition[code,100]//Image,decode//Image})&,320]
