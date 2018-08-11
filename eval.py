#coding=utf-8
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import time
import codecs
import sys
import os
from load_data import prepare_test_data,saving_answer_pre_sen


def write_answer_file(sentences,gold,file_name):
    now_time=str(time.asctime( time.localtime(time.time()) ))
    #base_path=sys.path[0]
    file=codecs.open(now_time,'w','utf-8')
    first_line=''
    for k,v in file_name.items():
        first_line+=k
        first_line+=': '
        first_line+=str(v)
        first_line+='   '
    first_line+='\n'
    file.write(first_line)
    test_nsen_tags=[ sen[1] for sen in sentences]
    gold_nsen_words=[sen[0] for sen in gold]
    for sen_tag,gold_word in zip(test_nsen_tags,gold_nsen_words):
        for (word,tag) in zip(gold_word,sen_tag):
            file.write(word)
            file.write('/')
            file.write(tag)
            file.write(' ')
        file.write('\n')
    file.close()


def score_dev(all_dict,tuple_list):#tuple is (list_of_predict_tag,list_of_true_tag)
    def col_sum(mat,col):
        sum_list=[mat[row][col] for row in range(mat.shape[0])]
        return sum(sum_list)

    def rows_sum(mat,row):
        row_list=[mat[row][col] for col in range(mat.shape[1])]
        return sum(row_list)

    def if_all_zeros(matrix,i):
        for j in matrix[i]:
            if j != 0. :
                return False
        else:
            return True


    tag_to_ix=all_dict['tag_to_ix']
    ix_to_tag=all_dict['ix_to_tag']
    dim=len(all_dict['tag_to_ix'])
    matrix=np.zeros((dim,dim),dtype=float)#行列分别对应着tagtoix中的顺序
    com_para=np.zeros(dim)
    for sen_list in tuple_list:
        pre_tag_ix_list= sen_list[0]
        true_tag_ix_list=sen_list[1]
        # 填充矩阵
        for pre_ix,true_ix in zip(pre_tag_ix_list,true_tag_ix_list):
            matrix[true_ix][pre_ix]+=1
    dim_real=0#本次devset包含的标签
    for i in range(len(matrix)):
        if not if_all_zeros(matrix,i):
            dim_real+=1
    #scoring
    score_dict={}#key 表示 tag，value为tupel(precision,recall,f)
    total_pre=[]
    total_re=[]
    total_f=[]
    for ix in range(len(matrix)):#对于每一行，即对每个tag
        if col_sum(matrix,ix) !=0 :
            pre=matrix[ix][ix]/col_sum(matrix,ix)

        else:
            pre=0

        if rows_sum(matrix,ix)!=0:

            recall=matrix[ix][ix]/rows_sum(matrix,ix)
        else:
            recall=0
        f=(pre+recall)/2
        total_f.append(f)
        total_re.append(recall)
        total_pre.append(pre)
        score_dict[ix_to_tag[ix]]=(pre,recall,f)
    #print (dim_real)
    score_dict['AVE']=(sum(total_pre)/dim_real,sum(total_re)/dim_real,sum(total_f)/dim_real)
    return score_dict


def plot_dev(tuple_list,epoch,interval):
    # pre_list=tuple_list[0]
    # recall_list=tuple_list[1]
    # f_list=tuple_list[2]
    pre_list=[]
    recall_list=[]
    f_list=[]
    for ix, i in enumerate(tuple_list):
        pre_list.append(i[0])
        recall_list.append(i[1])
        f_list.append(i[2])
    x=np.arange(interval,epoch+1,interval)
    assert len(x)==len(pre_list)
    plt.plot(x, pre_list,label='Precision',linewidth=3,color='r',marker='o', markerfacecolor='black',markersize=6)
    plt.plot(x, recall_list,label='Recall',linewidth=3,color='g',marker='o', markerfacecolor='black',markersize=6)
    plt.plot(x, f_list, label='F1', linewidth=3, color='blue', marker='o', markerfacecolor='black', markersize=6)
    plt.title('dev test result')
    plt.xlabel('epoch')
    plt.ylabel('recognization rate')
    plt.legend()
    plt.show()
    return


def eval_dev(all_dict,emb_model,model,dev_dataloader,if_cuda):#dataloader 每次输出的是一个tuplelist对，要进行转化
    '''one_hot_dict = all_dict['onehotdict']
    ix_to_tag = all_dict['ix_to_tag']
    word_to_embed = all_dict['word_to_embed']
    word_to_ix=all_dict['word_to_ix']'''
    result_tuple=[]
    if if_cuda:
        emb_model.cuda()
    for tuple in dev_dataloader:# list of word_embeding, a list of tags

        embeding=tuple[0][0]
        tag=tuple[0][1].view(1,-1)[0]

        # words_one_hot=[one_hot_dict[w] if w in one_hot_dict.keys() else one_hot_dict['UNK'] for w in row_word]
        # embeds_sen=[emb_model.get_embedding(one_hot_code.cuda() if if_cuda else one_hot_code).view(1,-1) for one_hot_code in words_one_hot]
        # embed_cated = torch.cat(embeds_sen, 0)
        #dev test
        result_tuple.append((model(embeding)[1],[t for t  in tag.int().numpy()]))#result tag list
    return score_dev(all_dict,result_tuple)


def eval_all(gold,test_dataloader, all_dict,file_name,emb_model,model):
    #one_hot_dict=all_dict['onehotdict']
    ix_to_tag=all_dict['ix_to_tag']
    word_to_embed=all_dict['word_to_embed']
    #sentence_list,gold=prepare_test_data(path)
    #将sentence转化成onehot，记录双向字典，
    #onehot生成词向量，通过lstm
    # prepared_test = []
    # for sen, tags in gold:
    #     one_hot_sen = [one_hot_dict[w] if w in one_hot_dict.keys() else one_hot_dict['UNK'] for w in sen]
    #     embeding_sen = [emb_model.get_embedding(code).view(1, -1) for code in one_hot_sen]
    #     embed_cated = torch.cat(embeding_sen, 0)
    #     #tag_tensors = [torch.tensor(tag_to_ix[t]).float().view(1, -1) for t in tags]
    #     #tag_cated = torch.cat(tag_tensors, 0)
    #     prepared_test.append((embed_cated, [w for w in sen]))
    #test_dataset=my_data_set(prepared_test)
    #test_datalaoder=Data.DataLoader(test_dataset,shuffle=True,collate_fn=coll_fn)
    all_result=[]
    for tuple in test_dataloader:
        embeding = tuple[0][0]
        gold_tag = tuple[0][1]
        all_result.append(saving_answer_pre_sen(model(embeding), ix_to_tag, word_to_embed, gold_tag))
    #write_answer_file(all_result,gold,file_name) Y
    score(gold, all_result,all_dict,file_name)


def score(gold,result,all_dicts,file_name):

    new_tag_set=[]
    tag_to_ix=all_dicts['tag_to_ix']
    # for gsen in gold:
    #     for g_tag in gsen[1]:
    #         if not tag_to_ix.__contains__(g_tag):
    #             new_tag_set.append(g_tag)
    # for tag in new_tag_set:
    #     assert tag_to_ix.__contains__(tag) is False
    #     tag_to_ix[tag]=len(tag_to_ix)

    ix_to_tag_new=all_dicts['ix_to_tag']
    # for tag, ix in tag_to_ix.items():
    #     assert not ix_to_tag_new.__contains__(ix)
    #     ix_to_tag_new[ix] = tag

    matrix=np.zeros((len(tag_to_ix.keys()),len(tag_to_ix.keys())))#混淆矩阵

    for rsen,gsen in zip(result,gold):#对每一句话
        for t_tag,g_tag in zip(rsen[1],gsen[1]):#对每一个标签
            matrix[tag_to_ix[g_tag] if g_tag in tag_to_ix else tag_to_ix['UNK']][tag_to_ix[t_tag] if t_tag in tag_to_ix else tag_to_ix['UNK']]+=1


    n = len(matrix)
    p_list=[]
    r_list=[]
    f1_list=[]
    now_time = str(time.asctime(time.localtime(time.time())))
    base_path = sys.path[0]
    answer_file = codecs.open(os.getcwd() + '/result_mini_plants/' + 'eval_data'+now_time.replace(':','_'), 'w', 'utf-8')

    #模型信息
    first_line = ''
    for k, v in file_name.items():
        first_line += k
        first_line += ': '
        first_line += str(v)
        first_line += '   '
    first_line += '\n'
    answer_file.write(first_line)
    for i in range(n):
        rowsum, colsum = sum(matrix[i]), sum(matrix[r][i] for r in range(n))
        w_string=''
        if rowsum!=0 and colsum!=0:
            p=matrix[i][i] / float(colsum)
            r=matrix[i][i] / float(rowsum)
            f1=(p+r)/2
            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)
            w_string+=ix_to_tag_new[i]+' '+str(rowsum)+' precision: ' + str(p) + ' recall: ' + str(r)+' f1 rate: ' +str(f1)
            print (ix_to_tag_new[i],' ',rowsum,'  precision: %s' % p, 'recall: %s' % r,' f1 rate: %s' %f1)

        else:
            w_string += ix_to_tag_new[i] + ' ' + str(rowsum) + ' precision: ' + str(0) + ' recall: ' + str(0) + ' f1 rate: ' + str(0)
            print (ix_to_tag_new[i],' ',rowsum,'  precision: %s' % 0, 'recall: %s' % 0,' f1 rate: %s' %0)
            #p_list.append(0)
            #r_list.append(0)
            #f1_list.append(0)
        answer_file.write(w_string+'\n')
    print ('total precision: ',sum(p_list)/float(len(p_list)),' recall: ',sum(r_list)/float(len(r_list)),' f1 rate: ',sum(f1_list)/float(len(f1_list)))
    answer_file.write('total precision: '+str(sum(p_list)/float(len(p_list)))+' recall: '+str(sum(r_list)/float(len(r_list)))+' f1 rate: '+str(sum(f1_list)/float(len(f1_list))))
    answer_file.close()
    print('file_close')

if __name__=='__main__':
    # p=np.arange(0,1,0.1)
    # r=np.arange(0,1,0.1)
    # f=np.arange(0,1,0.1)
    # plot_dev((p,r,f),10,1)
    now= str(time.asctime(time.localtime(time.time()))).replace(':','_')
    print(now)
    answer_file = codecs.open(os.getcwd() + '\\result_mini_plants\\' + 'eval_data ' + now, 'w', 'utf-8')
