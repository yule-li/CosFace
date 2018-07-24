## Recent Update
```2018.07.04```: I achieved a better accuracy(99.2%,[trained model](https://pan.baidu.com/s/1c7bPoM_hGvkzp5Tunu_ivg)) on LFW. I did some modification as bellow:
- Align webface and lfw dataset to ```112x112``` using [insightface align method](https://github.com/deepinsight/insightface/blob/master/src/align/align_lfw.py)
- Set a bigger margin parameter (```0.35```) and a higher feature embedding demension (```1024```)
- Use the clean dataset and the details can be seen [this](https://github.com/happynear/FaceVerification/issues/30)
## CosFace
This project is aimmed at implementing the CosFace described by the paper [**CosFace: Large Margin Cosine Loss for Deep Face Recognition**](https://arxiv.org/pdf/1801.09414.pdf). The code can be trained on [CASIA-Webface](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) and the best accuracy [LFW](http://vis-www.cs.umass.edu/lfw/) is 98.6%. The result is lower than reported by paper(99.33%), which may be caused by sphere network implemented in tensorflow. I train the sphere network implemented in tensorflow using the softmax loss and just obtain the accuracy 95.6%, which is more lower than caffe version(97.88%).

## Preprocessing
I supply the preprocessed dataset in baidu pan:[CASIA-WebFace-112X96](https://pan.baidu.com/s/160RN84j_79TnktKZmzakfw),[lfw-112X96](https://pan.baidu.com/s/1fkH9xR5Z0inxTP7Maae2KQ). You can download and unzip them to dir ```dataset```.

If you want to preprocess the dataset by yourself, you can refer to [sphereface](https://github.com/wy1iu/sphereface/tree/0056a7d27d05f2815a276cb26471f0348d6dd8da#installation).


## Train
```./train.sh``` 

## Test
Modify the ```MODEL_DIR``` in ```test.sh``` and run ```./test/sh```.

If you do not want to train your model, you can download my [trained model](https://pan.baidu.com/s/1ouQA2PXz1hp7Uz_uhsyMdw) and unzip it to ```models``` dir.

## Reference
- [facenet](https://github.com/davidsandberg/facenet)
