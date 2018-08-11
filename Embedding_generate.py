#coding=utf-8
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import codecs
import torch.optim as optim
import torch.utils.data as Data
import jieba
from my_dataset import Embeding_data_set
PATH='train_mini_plants.txt'
CONTEXT=2


class myEmbedding(nn.Module):
    def __init__(self,v_size,d_size):
        super(myEmbedding,self).__init__()
        print ('compiling...')
        self.v_size=v_size
        self.d_size=d_size
        #self.if_cuda=if_cuda
        self.embed_layer=nn.Linear(v_size,d_size,False)
        self.out_layer=nn.Linear(d_size,v_size,False)

    def forward(self,inputs):
        #assert len(input) == self.v_size
        out_list=[]
        for input in inputs:
            embds=self.embed_layer(input)

            out_list.append(embds.view(1,-1))

        cat_tensor=torch.cat(out_list,0)
        ave=torch.mean(cat_tensor,0)
        out = self.out_layer(ave)
        out=autograd.Variable(out,requires_grad=True).view(1,-1)
        #log_probs=F.log_softmax(out)
        #loss=nn.CrossEntropyLoss(out,target)
        return out

    def get_embedding(self,input_one_hot):
        return self.embed_layer(input_one_hot).detach()



def load_data(PATH):
    print ('loading data...')
    file=codecs.open(PATH,'r','utf-8')
    all_list=[]
    for line in file :
        sen_words=jieba.cut(line.rstrip())
        all_list.append(' '.join(sen_words).split(' '))
    return all_list


def make_dict(list_of_words):
    words_dict={}
    for sen in list_of_words:
        for word in sen:
            if words_dict.__contains__(word):
                pass
            else:
                words_dict[word]=len(words_dict)
    #add unk
    words_dict['UNK']=len(words_dict)
    return words_dict


def build_one_hot(dict):#建立词编码与张量的字典
    #size= sentence*word*size
    dim=len(dict)
    one_hot_dict={}
    word_tensor_list=[]
    for word in dict.keys():
        word_tensor=torch.zeros(dim,dtype=torch.float)
        word_tensor[dict[word]]=1
        if not one_hot_dict.__contains__(word) :
            one_hot_dict[word]=word_tensor
        else:
            pass

    # cat all tensor
    #all_2d_tensor=torch.cat(word_tensor_list,0)
    return one_hot_dict# one hot model


def build_data(list_of_sen):
    data=[]
    for sen_ix in range(len(list(list_of_sen))):
        for i in range(2,len(list_of_sen[sen_ix])-2):
            context=[list_of_sen[sen_ix][i-2],list_of_sen[sen_ix][i-1],list_of_sen[sen_ix][i+1],list_of_sen[sen_ix][i+2]]
            target=list_of_sen[sen_ix][i]

            data.append((target,context))

    return data

def index_to_one_hot(ix,total_number):
    a=torch.zeros([total_number])
    a[ix]=1
    return a


def word_to_tensor(data,word_dict,if_cuda=False):
    tensor_data=[]
    for tar,context in data:
        tar_new=word_dict[tar]
        context_new=[word_dict[w]  for w in context]
        tensor_data.append((tar_new,context_new))
    return tensor_data


def build_embed_dict(path):#used only if load pretrained embedding
    all_list = load_data(path)
    print ('building dicts...')
    word_dict = make_dict(all_list)  # word_to_idx
    #ix_to_word = {ix: w for (w, ix) in word_dict.items()}
    #one_hot_dict = build_one_hot(word_dict)  # word_ to _one hot
    print ('one-hot-code build')
    #one_hot_to_word = {hot: w for (w, hot) in one_hot_dict.items()}
    # data_word=build_data(all_list)
    print ('preparing Embedding data...')
    # data = []
    # for sen_ix in range(len(list(all_list))):
    #     for i in range(2, len(all_list[sen_ix]) - 2):
    #         context = [all_list[sen_ix][i - 2], all_list[sen_ix][i - 1], all_list[sen_ix][i + 1],
    #                    all_list[sen_ix][i + 2]]
    #         target = all_list[sen_ix][i]
    #
    #         data.append((target, context))

    #tensor_list_of_tuple = word_to_tensor(data, word_dict, one_hot_dict)
    return word_dict



def build_model(path,emb_dim,cuda):
    all_list = load_data(path)
    if_cuda=cuda
    print ('building dicts...')
    word_dict = make_dict(all_list) #word_to_idx,start from 0
    #ix_to_word = {ix: w for (w, ix) in word_dict.items()}
    #one_hot_dict = build_one_hot(word_dict)#word_ to _one hot
    #error no enough memory for 30000 words
    print ('one-hot-code build')
    #one_hot_to_word = {hot: w for (w, hot) in one_hot_dict.items()}
    # data_word=build_data(all_list)
    print ('preparing Embedding data...')
    data = []
    for sen_ix in range(len(list(all_list))):
        for i in range(2, len(all_list[sen_ix]) - 2):
            context = [all_list[sen_ix][i - 2], all_list[sen_ix][i - 1], all_list[sen_ix][i + 1],
                       all_list[sen_ix][i + 2]]
            target = all_list[sen_ix][i]

            data.append((target, context))

    #del all_list

    tensor_list_of_tuple = word_to_tensor(data,word_dict,False)
    #del data

    my_embedding_dataset=Embeding_data_set(tensor_list_of_tuple,len(word_dict),if_cuda)
    #embeding_dataloader=Data.DataLoader(my_embedding_dataset,collate_fn=coll_fn,num_workers=0)

    mymodel = myEmbedding(len(word_dict), emb_dim)
    total_words_number=len(word_dict)
    if if_cuda:
        mymodel=mymodel.cuda()

    if if_cuda:
        loss = nn.CrossEntropyLoss().cuda()
    else:
        loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mymodel.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in range(100):
        print (epoch)
        mymodel.train()
        for ix in range(len(my_embedding_dataset)):
            label,inputs=my_embedding_dataset.getitem(ix)
            # mymodel.zero_grad()
            label = torch.tensor([label])
            if if_cuda:
                label=label.cuda()
            #inputs = item[0][1]
            #input_tensors=[ index_to_one_hot(i,total_words_number).cuda() if if_cuda else index_to_one_hot(i,total_words_number) for i in inputs]

            out = mymodel(inputs)

            loss_data = loss(out, label)
            # print type(loss_data)

            optimizer.zero_grad()

            loss_data.backward()

            optimizer.step()

    torch.save(mymodel, 'Embedding.pkl')
    print ('saved embedding')
    return mymodel, word_dict

if __name__=='__main__':
    build_model(PATH)
'''
if __name__=='__main__':
    Train=True
    all_list=load_data(PATH)
    print 'building dicts...'
    word_dict=make_dict(all_list)
    ix_to_word={ix:w for (w,ix) in word_dict.items()}
    one_hot_dict=build_one_hot(word_dict)
    print 'onr-hot-code build'
    one_hot_to_word={hot:w for (w,hot) in one_hot_dict.items()}
    #data_word=build_data(all_list)
    print 'preparing Embedding data...'
    data=[]
    for sen_ix in range(len(list(all_list))):
        for i in range(2,len(all_list[sen_ix])-2):
            context=[all_list[sen_ix][i-2],all_list[sen_ix][i-1],all_list[sen_ix][i+1],all_list[sen_ix][i+2]]
            target=all_list[sen_ix][i]

            data.append((target,context))

    tensor_list_of_tuple=word_to_tensor(data,word_dict,one_hot_dict)

    mymodel=myEmbedding(len(word_dict),100)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mymodel.parameters(), lr=0.001, weight_decay=1e-4)

    if Train:
        for epoch in range(100):
            print epoch
            mymodel.train()
            for item in tensor_list_of_tuple:
                #mymodel.zero_grad()

                label=torch.tensor([item[0]])
                inputs=item[1]

                out=mymodel.forward(inputs)

                loss_data=loss(out,label)
                #print type(loss_data)

                optimizer.zero_grad()

                loss_data.backward()

                optimizer.step()



        torch.save(mymodel.state_dict(), 'Embedding.pkl')
        print 'saved!'
    else:
        mymodel.load_state_dict(torch.load('Embedding.pkl'))

    for w in word_dict.keys():
        #test_word='UNK'
        #ix=word_dict[test_word]
        tese_one_hot=one_hot_dict[w]
        test_embedding=mymodel.get_embedding(tese_one_hot)
        print 'test_word: ', w
        print 'tese_one_hot: ',tese_one_hot
        print 'test_embedding: ',test_embedding

'''
