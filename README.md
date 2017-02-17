# Welcome to CognitoNet.

*
17/02/2017 Update:
This project is no longer in active development. The project has been a lot of fun to develop, but the world has now moved on. For serious machine learning I would suggest the new machine learning tools in Mathematica v11, which I believe is based on CAFFE, a well respected deep learning library. I am using TensorFlow in Python which I am having good experiences with. I am also recommending for people seriously interested in machine learning to be multilingual and use several tools. The downside is the overhead in learning other languages and tools. However many research papers leave many technical details unspecified and where they publish a working reference model this can be invaluable in clarifying these details (and debugging).

*To migrate models from CognitoNet to other systems I have found a useful strategy is to respecify the model in the new framework, and use the JSON file format to transfer weights from the CognitoNet model to the new framework.

CognitoNet is a Mathematica implementation of Convolutional Neural Nets.

Objectives of the project are to make neural networks:

 - Accessible
 - Interactive
 - Flexible
 - Extensible
 - Efficient

in descending order of priority.

The package extends the machine learning capabilities of Mathematica (introduced in version 10) to multi layer convolutional nets. The package has been influenced by other popular machine learning systems, including Caffe and Torch, and Alex Krizhevsky's Convnet. However, it differs from these packages in that by using Mathematica it supports a highly interactive environment facilitating experimentation. It also has an unusually small code base (around 800 lines). Although other GPU based machine learning systems are faster, it is an objective that it should be sufficiently efficient to solve interesting and challenging problems such as CIFAR-10 and face detection.

The main application area so far is computer vision. In principle it could be of use in any other area where a feed forward neural network is applicable.

Use of this package assumes some familiarity with the concepts of a convolutional neural network, eg you have been following Geoffrey Hinton's Coursera course on neural networks (https://www.coursera.org/), or [Nando de Freitas Deep Learning Course](https://www.youtube.com/watch?v=PlhFWT7vAEw&list=PLE6Wd9FR--EfW8dtjAuPoTuPcqmOV53Fu&index=16) or you have equivalent knowledge.

Example neural nets include:

 - [MNIST](https://github.com/jfrancis71/CognitoNet/wiki/MNIST)
 - [CIFAR-10](https://github.com/jfrancis71/CognitoNet/wiki/CIFAR-10-Example)
 - [FaceNet](https://github.com/jfrancis71/CognitoNet/wiki/Face-Detection) (Face and gender recognition)
 
    YouTube Video Demonstration: <https://www.youtube.com/watch?v=TdRtUnSppB0>
 - Self Driving Robot (Lego EV3)

   YouTube Video Demonstration: <https://www.youtube.com/watch?v=DCad82UdDFA> 

 - "No Entry" Sign Localization

   YouTube Video Demonstration: https://www.youtube.com/watch?v=LCgRAmG56Uo 
