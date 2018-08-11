# LSTM-crf-model-Chinese-NLP
This is a model based on LSTM and CRF, predicting the tag of Chinese words.
row data is simple Chinese paragraphs.
Some basic configuration us in LSTM.py

## prerequisite
python 3.6
pytorch 0.4
jieba
cuda 9.2
numpy

## model
using word embedding to transform sqarse vector of a word to density word embedding vector.
bidirectional LSTM is the first layer of model, which output is the input of CRF layer(the second layer)
The output of CRF layer is the final tag probability of each tags.

## setup
See LSTM.py 
