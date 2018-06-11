# DE-CNN-TF
The tensorflow version of Double-Embeddings-and-CNN-based-Sequence-Labeling-for-Aspect-Extraction

Original version is implemented by pytorch, the link is (https://github.com/howardhsu/Double-Embeddings-and-CNN-based-Sequence-Labeling-for-Aspect-Extraction).

All code of this version are tested under python 2.7.14 + tensorflow 1.3.0

Previous 6 steps follow Pytorch version. The main differences are train_tf.py and evaluation_tf.py

Step 7: Train the laptop model
```
python script/train_tf.py
```
Train the restaurant model
```
python script/train_tf.py --domain restaurant
```

Step 8: Evaluate Laptop dataset
```
python script/evaluation_tf.py
```
Evaluate Restaurant dataset
```
python script/evaluation_tf.py --domain restaurant
```

