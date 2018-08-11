#coding=utf-8
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
def to_scalar(var):  # var是Variable,维度是１
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)#返回一个数对，后一个为一个tensor，
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w if w in to_ix.keys() else 'UNK']  for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec): #vec是1*5, type是Variable

    max_score = vec[0, argmax(vec)]#得到最大，max_score is the largest pro in the tag set a float num
    max_score=max_score.view(1)
    #max_score维度是１，　max_score.view(1,-1)维度是１＊１，max_score.view(1, -1).expand(1, vec.size()[1])的维度是１＊５
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1]) # vec.size()维度是1*5 每个元素均等于maxscore的值
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))#为什么指数之后再求和，而后才log呢 size=1*5