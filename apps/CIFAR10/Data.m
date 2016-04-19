(* ::Package:: *)

(*
   Ref: CIFAR 10 - 
      Learning Multiple Layers Of Features From Tiny Images
      Krizhevsky - 2009
   http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
*)


CIFARReadFile[filename_String] := {
   Map[First,Partition[BinaryReadList[filename],3073]],
   Map[Function[{flat},Partition[flat,32]],Map[Partition[#,1024]&,Map[Rest,Partition[BinaryReadList[filename],3073]]],{2}]/256.
};


CIFARFilesForTraining = FileNames[$CNDataDir<>"\\CIFAR10\\data*.bin"];


TrainingLabels = Flatten[
   Map[CIFARReadFile,CIFARFilesForTraining]
   [[All,1]]];


TrainingImages = Flatten[
   Map[CIFARReadFile,
      CIFARFilesForTraining][[All,2]],1];


TestLabels = CIFARReadFile[$CNDataDir<>"\\CIFAR10\\test_batch.bin"][[1]];


TestImages = CIFARReadFile[$CNDataDir<>"\\CIFAR10\\test_batch.bin"][[2]];


CIFAR10CategoryIndexToLabelMap = Table[k,{k,0,9}];


CIFAR10CategoryIndexToTxtLabelMap = {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
