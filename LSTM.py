#coding=utf-8
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from load_data import *
import numpy as np
from nn import BiLSTM_CRF
from utils import *
from my_dataset import my_data_set,coll_fn
from Embedding_generate import build_model,myEmbedding,build_embed_dict,index_to_one_hot
from eval import eval_all,eval_dev,plot_dev

#torch.backends.cudnn.benchmark=True
'''
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
'''

'''
def eval_all(path, all_dict):
    sentence_list,gold=prepare_test_data(path)
    #将sentence转化成onehot，记录双向字典，
    #onehot生成词向量，通过lstm
    prepared_test = []
    for sen, tags in gold:
        one_hot_sen = [all_dict[one_hot_dict][w] if w in one_hot_dict.keys() else all_dict[one_hot_dict]['UNK'] for w in sen]
        embeding_sen = [emb_model.get_embedding(code).view(1, -1) for code in one_hot_sen]
        embed_cated = torch.cat(embeding_sen, 0)
        #tag_tensors = [torch.tensor(tag_to_ix[t]).float().view(1, -1) for t in tags]
        #tag_cated = torch.cat(tag_tensors, 0)
        prepared_test.append((embed_cated, [w for w in sen]))
    #test_dataset=my_data_set(prepared_test)
    #test_datalaoder=Data.DataLoader(test_dataset,shuffle=True,collate_fn=coll_fn)
    all_result=[]
    for tuple in prepared_test:
        test_sen=tuple[0]
        #gold_tag=tuple[0][1]
        all_result.append(saving_answer_pre_sen(model(test_sen),ix_to_tag,word_to_embed,tuple[1]))
    write_answer_file(all_result,gold,file_name)
    score(gold, all_result)'''

'''
    answer_all_sen=[]#each element is a list(a sentence) which contais tuples of words and tags
    for sen in sentence_list:
        varible_sen=prepare_sequence(sen,all_dict['word_to_ix'])
        answer_all_sen.append(saving_answer_pre_sen(model(varible_sen),all_dict['ix_to_tag'],all_dict['ix_to_word']))
    write_answer_file(answer_all_sen,gold,file_name)
    score(gold,answer_all_sen)'''

# def score(gold,result,all_dicts):
#     '''total=0
#     correct=0
#     for (r_sen,g_sen) in zip(result,gold):
#         assert len(r_sen)==2 and len(g_sen)==2
#         #assert r_sen[0]==g_sen[0]
#         for r_t,g_t in zip(r_sen[1],g_sen[1]):
#             total+=1
#             if r_t==g_t:
#                 correct+=1
#             else:
#                 pass
#     print correct/float(total)'''
#
#
#     new_tag_set=[]
#     for gsen in gold:
#         for g_tag in gsen[1]:
#             if not tag_to_ix.has_key(g_tag):
#                 new_tag_set.append(g_tag)
#     for tag in new_tag_set:
#         assert tag_to_ix.has_key(tag) is False
#         tag_to_ix[tag]=len(tag_to_ix)
#
#     ix_to_tag_new={}
#     for tag, ix in tag_to_ix.items():
#         assert not ix_to_tag_new.has_key(ix)
#         ix_to_tag_new[ix] = tag
#
#     matrix=np.zeros((len(tag_to_ix.keys()),len(tag_to_ix.keys())))#混淆矩阵
#
#     for rsen,gsen in zip(result,gold):#对每一句话
#         for t_tag,g_tag in zip(rsen[1],gsen[1]):#对每一个标签
#             matrix[tag_to_ix[g_tag]][tag_to_ix[t_tag]]+=1
#
#     '''for i in range(len(matrix)):
#         print ix_to_tag_new[i],'   ',matrix[i]'''
#     n = len(matrix)
#     p_list=[]
#     r_list=[]
#     f1_list=[]
#     now_time = str(time.asctime(time.localtime(time.time())))
#     base_path = sys.path[0]
#     answer_file = codecs.open(base_path + '/result_mini_plants/' + 'eval_data'+now_time, 'w', 'utf-8')
#
#     #模型信息
#     first_line = ''
#     for k, v in file_name.items():
#         first_line += k
#         first_line += ': '
#         first_line += str(v)
#         first_line += '   '
#     first_line += '\n'
#     answer_file.write(first_line)
#     for i in range(n):
#         rowsum, colsum = sum(matrix[i]), sum(matrix[r][i] for r in range(n))
#         w_string=''
#         if rowsum!=0 and colsum!=0:
#             p=matrix[i][i] / float(colsum)
#             r=matrix[i][i] / float(rowsum)
#             f1=(p+r)/2
#             p_list.append(p)
#             r_list.append(r)
#             f1_list.append(f1)
#             w_string+=ix_to_tag_new[i]+' '+str(rowsum)+' precision: ' + str(p) + ' recall: ' + str(r)+' f1 rate: ' +str(f1)
#             print (ix_to_tag_new[i],' ',rowsum,'  precision: %s' % p, 'recall: %s' % r,' f1 rate: %s' %f1)
#
#         else:
#             w_string += ix_to_tag_new[i] + ' ' + str(rowsum) + ' precision: ' + str(0) + ' recall: ' + str(
#                 0) + ' f1 rate: ' + str(0)
#             print (ix_to_tag_new[i],' ',rowsum,'  precision: %s' % 0, 'recall: %s' % 0,' f1 rate: %s' %0)
#             #p_list.append(0)
#             #r_list.append(0)
#             #f1_list.append(0)
#         answer_file.write(w_string+'\n')
#     print ('total precision: ',sum(p_list)/float(len(p_list)),' recall: ',sum(r_list)/float(len(r_list)),' f1 rate: ',sum(f1_list)/float(len(f1_list)))
#     answer_file.write('total precision: '+str(sum(p_list)/float(len(p_list)))+' recall: '+str(sum(r_list)/float(len(r_list)))+' f1 rate: '+str(sum(f1_list)/float(len(f1_list))))
#     answer_file.close()
#     print('file_close')


'''
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        self.dropout=nn.Dropout(p=0.2)

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
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),#返回一个tuple，两个元素分别为tensor，三维张量
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))
    # 预测序列的得分

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)

        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)  # 初始状态的forward_var，随着step t变化

        # Iterate through the sentence
        for feat in feats: #feat的维度是５,feat 是每个单词的各标签预测概率
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):#对下一个标签进行预测
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size) #维度是1*5

                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1) #维度是１＊５
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
        embeds = self.word_embeds(sentence)

        embeds=self.dropout(embeds)

        embeds = embeds.unsqueeze(1)

        lstm_out, self.hidden = self.LSTM(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)

        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats


    # 得到gold_seq tag的score
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags]) #将START_TAG的标签３拼接到tag序列上

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
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
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
        gold_score = self._score_sentence(feats, tags)

        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq, sentence
'''

#初始化
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 25
HIDDEN_DIM = 50
TRAIN=True
PRETRAIN_EMB=True
if_cuda=False
PATH='train_mini_plants_half.txt'
DEV_PATH='dev_mini_plants.txt'
TEST_PATH='test_mini_plants.txt'
EPOCH=100
INTERVAL=10
file_name={
    'EMBEDDING_DIM':EMBEDDING_DIM,
    'HIDDEN_DIM':HIDDEN_DIM,
    'EPOCH':EPOCH
}
# Make up some training data
'''training_data = [("the wall street journal reported today that apple corporation made money".split(), "B I I I O O O B I O O".split()),
                 ("georgia tech is a university in georgia".split(), "B I O O O O B".split())]'''


if not PRETRAIN_EMB:
    print ('start train Embedding layer...')
    emb_model,word_dict=build_model(PATH,EMBEDDING_DIM,if_cuda)
    print ('train Embedding layer finish')
else:
    print ('loading Embedding layer...')
    emb_model=torch.load('Embedding.pkl')
    word_dict=build_embed_dict(PATH)
emb_model.cpu()
emb_model.eval()

training_data=prepare_training_data(PATH)
dev_data=prepare_training_data(DEV_PATH)
#create dicts
#build word_to_embeding
word_to_ix = word_dict
words_number=len(word_to_ix)
'''
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
#添加unk
word_to_ix['UNK']=len(word_to_ix)'''
ix_to_word={}
for word,ix in word_to_ix.items():
    assert not ix_to_word.__contains__(ix)
    ix_to_word[ix]=word
tag_to_ix,tag_name_dict=get_tag_dict('jieba_tagset')
ix_to_tag={}
for tag,ix in tag_to_ix.items():
    assert not ix_to_tag.__contains__(ix)
    ix_to_tag[ix]=tag
word_to_embed={}
for w in word_to_ix.keys():
    word_to_embed[w]=emb_model.get_embedding(index_to_one_hot(word_to_ix[w],len(word_to_ix)))
embed_to_word={
    w:v for (v,w) in word_to_embed.items()
}

all_dict={
    'word_to_ix':word_to_ix,
    'ix_to_word':ix_to_word,
    'tag_to_ix':tag_to_ix,
    'ix_to_tag':ix_to_tag,
    #'onehotdict':one_hot_dict,
    'word_to_embed':word_to_embed
}

# 将trainingdata里面的所有汉字转化为one hot 编码，然后生成词向量，保存到dataloader中
prepared_data=[]
for sen,tags in training_data:
    one_hot_sen=[index_to_one_hot(word_to_ix[w],words_number) if w in word_to_ix.keys() else index_to_one_hot(word_to_ix['UNK'],words_number) for w in sen]
    embeding_sen=[emb_model.get_embedding(code).view(1,-1) for code in one_hot_sen]
    embed_cated=torch.cat(embeding_sen,0)
    tag_tensors=[torch.tensor(tag_to_ix[t] if t in tag_to_ix.keys() else tag_to_ix['UNK']).float().view(1,-1) for t in tags]
    tag_cated=torch.cat(tag_tensors,0)
    prepared_data.append((embed_cated,tag_cated))

prepared_dev_data=[]
for sen,tags in dev_data:
    one_hot_sen = [
        index_to_one_hot(word_to_ix[w], words_number) if w in word_to_ix.keys() else index_to_one_hot(word_to_ix['UNK'],
                                                                                                      words_number) for
        w in sen]
    embeding_sen=[emb_model.get_embedding(code).view(1,-1) for code in one_hot_sen]
    embed_cated=torch.cat(embeding_sen,0)
    tag_tensors=[torch.tensor(tag_to_ix[t] if t in tag_to_ix.keys() else tag_to_ix['UNK']).float().view(1,-1) for t in tags]
    tag_cated=torch.cat(tag_tensors,0)
    prepared_dev_data.append((embed_cated,tag_cated))

sentence_list,gold=prepare_test_data(TEST_PATH)
prepared_test_data=[]
for sen,tags in gold:
    one_hot_sen = [
        index_to_one_hot(word_to_ix[w], words_number) if w in word_to_ix.keys() else index_to_one_hot(word_to_ix['UNK'],
                                                                                                      words_number) for
        w in sen]
    embeding_sen = [emb_model.get_embedding(code).view(1, -1) for code in one_hot_sen]
    embed_cated = torch.cat(embeding_sen, 0)
    tag_tensors = [torch.tensor(tag_to_ix[t] if t in tag_to_ix.keys() else tag_to_ix['UNK']).float().view(1, -1) for t in tags]
    tag_cated = torch.cat(tag_tensors, 0)
    prepared_test_data.append((embed_cated, tag_cated))

assert prepared_data is not None
assert prepared_test_data is not None
# Embedding coding of one-hot code

#create dicts finish
#tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
train_dataset=my_data_set(prepared_data,if_cuda)
train_dataloader=Data.DataLoader(train_dataset,shuffle=False,collate_fn=coll_fn,num_workers=0)
#build dev dataset
dev_dataset=my_data_set(prepared_dev_data,if_cuda)
dev_dataloader=Data.DataLoader(dev_dataset,shuffle=True,collate_fn=coll_fn,num_workers=0)
test_dataset=my_data_set(prepared_test_data,if_cuda)
test_dataloader=Data.DataLoader(test_dataset,shuffle=False,collate_fn=coll_fn,num_workers=0)
#build model
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM,if_cuda)
if if_cuda:
    model=model.cuda()
del prepared_data,training_data
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
print ('总共单词个数为:',len(word_to_ix))
print ('总标签集:',len(tag_to_ix))
# Check predictions before training
'''
precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
precheck_tags = torch.LongTensor([tag_to_ix[t] for t in training_data[0][1]])
#print(model(precheck_sent))
#a=model(precheck_sent)
print_answer(model(precheck_sent),ix_to_tag,ix_to_word)
'''
# Make sure prepare_sequence from earlier in the LSTM section is loaded
plot_list=[]#list of tuple(pre,recall,f)
if TRAIN:
    for epoch in range(EPOCH):  # again, normally you would NOT do 300 epochs, it is toy data
        print ('epoch:',epoch)
        model.train()
        # for ix in range(len(train_dataset)):
        #     tuple=train_dataset.getitem(ix)
        for tuple in train_dataloader:
            sentence=tuple[0][0]# a list of embedding tensor
            tags=tuple[0][1]  #a list of char
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Variables of word indices.
            #sentence_in = prepare_sequence(sentence, word_to_ix)
            #targets = torch.LongTensor([tag_to_ix[t] for t in tags])

            # Step 3. Run our forward pass.
            neg_log_likelihood = model.neg_log_likelihood(sentence, tags)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            neg_log_likelihood.backward(retain_graph=False)
            optimizer.step()
        #dev test
        if (epoch+1)%INTERVAL == 0 :
            model.eval()
            plot_list.append(eval_dev(all_dict,emb_model,model,dev_dataloader,if_cuda).get('AVE'))
    torch.save(model.state_dict(), 'model_parms.pkl')
else:
    model.load_state_dict(torch.load('model_parms.pkl'))
# Check predictions after training
#precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
model.eval()
eval_all(gold,test_dataloader,all_dict,file_name,emb_model,model)
plot_dev(plot_list,EPOCH,INTERVAL)
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

#print_answer(model(precheck_sent),ix_to_tag,ix_to_word) #得分

#print(model(precheck_sent)[1]) #tag sequence.Variable(torch.randn((1,3))) for _ in range(5)]