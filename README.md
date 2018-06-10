# DE-CNN-TF
The Tensorflow version of Double-Embeddings-and-CNN-based-Sequence-Labeling-for-Aspect-Extraction
Original version is implemented by Pytorch, the link is (https://github.com/howardhsu/Double-Embeddings-and-CNN-based-Sequence-Labeling-for-Aspect-Extraction).

All code of this version are tested under python 2.7.14 + tensorflow 1.3.0

Step 5: Run prep_embedding.py to build numpy files for general embeddings and domain embeddings.
```
python script/prep_embedding.py
```

Step 6: Fill in out-of-vocabulary (OOV) embedding
```
./fastText/fasttext print-word-vectors data/embedding/laptop_emb.vec.bin < data/prep_data/laptop_emb.vec.oov.txt > data/prep_data/laptop_oov.vec

./fastText/fasttext print-word-vectors data/embedding/restaurant_emb.vec.bin < data/prep_data/restaurant_emb.vec.oov.txt > data/prep_data/restaurant_oov.vec

python script/prep_oov.py
```

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

