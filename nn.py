#coding=utf-8
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from utils import *
START_TAG = "<START>"
STOP_TAG = "<STOP>"


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,if_cuda):
        super(BiLSTM_CRF, self).__init__()
        #self.embedding_dim = embedding_dim
        self.if_cuda=if_cuda
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        #self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        self.dropout=nn.Dropout()

        self.LSTM = nn.LSTM(embedding_dim, hidden_dim //2 , num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        #transitions[i][j] means from tag[j]->tag[i]

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000#from other tags to start is -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000#start with end tag is -10000

        self.hidden = self.init_hidden()#is a tuple

    def init_hidden(self):
        if self.if_cuda:

            return (torch.randn(2, 1, self.hidden_dim//2 ).cuda(),#返回一个tuple，两个元素分别为tensor，三维张量
                    torch.randn(2, 1, self.hidden_dim//2 ).cuda())
        else:
            return (torch.randn(2, 1, self.hidden_dim // 2),  # 返回一个tuple，两个元素分别为tensor，三维张量
                    torch.randn(2, 1, self.hidden_dim // 2))

    # 预测序列的得分

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)

        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        #起始概率为1*tagsize的矩阵，将starttag为0，其余为-无穷大

        # Wrap in a variable so that we will get automatic backprop

        forward_var = autograd.Variable(init_alphas)  # 初始状态的forward_var，随着step t变化
        if self.if_cuda:
            forward_var=forward_var.cuda()

        # Iterate through the sentence
        for feat in feats: #feats为矩阵（单词个数*dim_size），feat为矩阵内部的每一个向量，
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):#对下一个标签进行预测
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size) #维度是1*5，data全部为feat[next_tag]的数值

                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1) #维度是１＊tagsize，表示其他标签转移到次标签的概率
                if self.if_cuda:
                    trans_score=trans_score.cuda()
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                #第一次迭代时理解：
                # trans_score所有其他标签到Ｂ标签的概率
                # 由lstm运行进入隐层再到输出层得到标签Ｂ的概率，emit_score维度是１＊５，5个值是相同的
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))# a list of tensor 1*5
            #print alphas_t[0].dim()
            #for item in alphas_t:
            #    print item.size()

            forward_var = torch.cat(alphas_t).view(1, -1)#到第（t-1）step时５个标签的各自分数 size 1*5
            #print forward_var.size()
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)

        return alpha

    # 得到feats
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        #embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        #embeds = self.word_embeds(sentence)

        #embeds=self.dropout(sentence)

        embeds = sentence.unsqueeze(1)

        lstm_out, self.hidden = self.LSTM(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_out=self.dropout(lstm_out)

        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats


    # 得到gold_seq tag的score
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        if self.if_cuda:
            score=score.cuda()
        start_tag_tensor=torch.tensor([self.tag_to_ix[START_TAG]]).view(-1,1).float()
        if self.if_cuda:
            start_tag_tensor=start_tag_tensor.cuda()

        tags = torch.cat([start_tag_tensor, tags]).long() #将START_TAG的标签３拼接到tag序列上

        for i, feat in enumerate(feats):
            #self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
            #feat[tags[i+1]], feat是step i 的输出结果，有５个值，对应B, I, E, START_TAG, END_TAG, 取对应标签的值

            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    # 解码，得到预测的序列，以及预测序列的得分
    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)#创建一个1*tagset的向量，dim为2
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0#起始到start tag的概率为0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)#起始概率值
        if self.if_cuda:
            forward_var=forward_var.cuda()
        for feat in feats:#feat是每一个单词的对应tag概率的向量，dim1 1*tagsize
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):#对于每一个标签
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag] #其他标签（B,I,E,Start,End）到标签next_tag的概率
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)

                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)#从step0到step(i-1)时5个序列中每个序列的最大score
            backpointers.append(bptrs_t) #bptrs_t有５个元素

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]#其他标签到STOP_TAG的转移概率
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):#从后向前走，找到一个best路径
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()# 把从后向前的路径正过来
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        #forward_score=log_sum_exp(feats)
        gold_score = self._score_sentence(feats, tags)
        answer=forward_score - gold_score

        return answer

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq, sentence
