(* ::Package:: *)

<<CNNeuralCore.m


Set::write


inp=Import["/Users/julian/tensorflow/lib/python2.7/site-packages/tensorflow/models/image/mnist/input.json"];


outp=Import["/Users/julian/tensorflow/lib/python2.7/site-packages/tensorflow/models/image/mnist/output.json"];


bias1=Import["/Users/julian/tensorflow/lib/python2.7/site-packages/tensorflow/models/image/mnist/conv1bias.json"];
weights1=Transpose[Import["/Users/julian/tensorflow/lib/python2.7/site-packages/tensorflow/models/image/mnist/conv1weights.json"][[All,All,1]],{2,3,1}];
bias2=Import["/Users/julian/tensorflow/lib/python2.7/site-packages/tensorflow/models/image/mnist/conv2bias.json"];
weights2=Transpose[Import["/Users/julian/tensorflow/lib/python2.7/site-packages/tensorflow/models/image/mnist/conv2weights.json"],{3,4,2,1}];
fc1b=Import["/Users/julian/tensorflow/lib/python2.7/site-packages/tensorflow/models/image/mnist/b_fc1.json"];
fc1W=Map[Flatten[Transpose[Partition[#,64]]]&,Transpose[Import["/Users/julian/tensorflow/lib/python2.7/site-packages/tensorflow/models/image/mnist/W_fc1.json"]]];fc2b=Import["/Users/julian/tensorflow/lib/python2.7/site-packages/tensorflow/models/image/mnist/b_fc2.json"];
fc2W=Transpose[Import["/Users/julian/tensorflow/lib/python2.7/site-packages/tensorflow/models/image/mnist/W_fc2.json"]];


tensorNet={
PadFilter[2],
Convolve2DToFilterBank[
Table[Convolve2D[bias1[[f]],weights1[[f]]],{f,1,32}]
],
ReLU,
MaxPoolingFilterBankToFilterBank,
PadFilterBank[2],
ConvolveFilterBankToFilterBank[Table[ConvolveFilterBankTo2D[
bias2[[o]],Table[weights2[[o,f]],{f,1,32}]]
,{o,1,64}]],
ReLU,
MaxPoolingFilterBankToFilterBank,
AdaptorFilterBankTo1D[ 64,7,7],
FullyConnected1DTo1D[fc1b, fc1W],
ReLU,
FullyConnected1DTo1D[fc2b, fc2W],
Softmax
};


net={
PadFilter[2],
Convolve2DToFilterBankInit[32,5],
ReLU,
MaxPoolingFilterBankToFilterBank,
PadFilterBank[2],
ConvolveFilterBankToFilterBankInit[64,32,5],
ReLU,
MaxPoolingFilterBankToFilterBank,
AdaptorFilterBankTo1D[ 64,7,7],
FullyConnected1DTo1DInit[64*7*7,1024],
ReLU,
FullyConnected1DTo1DInit[1024, 10],
Softmax
};
