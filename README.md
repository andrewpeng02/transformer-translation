# transformer-translation
Using Pytorch's nn.Transformer module to create an english to french neural machine translation model. Details in https://andrewpeng.dev/transformer-pytorch/.

# Training
To install the prerequisites into a conda environment, run
``` 
conda env create -f environment.yml
```
Install and extract the english-french dataset from http://www.manythings.org/anki/ into the data/raw folder. Then run process-tatoeba-data.py, preprocess-data.py, then train.py

# Inference
Run translate-sentence.py, which uses the transformer.pth model in /output. 
