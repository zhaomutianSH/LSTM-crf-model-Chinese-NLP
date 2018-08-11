#coding=utf-8
import jieba
import jieba.posseg as pseg
import codecs
import sys
import time
import json


def prepare_training_data(path,if_jieba=True):
    all_list=prepare_sentence(path,if_jieba)
    return all_list


def prepare_test_data(path):
    all_list=prepare_sentence(path)
    words_list=[]
    gold_tags_list=[]
    for (w,t) in all_list:
        words_list.append(list(w))
        gold_tags_list.append(list(t))
    return words_list,all_list
    #all_list structure
    #list  [tuple(list[row_word,row_word],list[tag,tag])]
    #word list structure
    #list [list[row words]]

def load_sentances(path):
    lines=codecs.open(path,'r','utf-8')
    answer=[]
    for line in lines:
        new_line=''
        line=line.rstrip().replace(' ','')
        word_flag=pseg.cut(line)
        for word , tag in word_flag:
            answer.append((word,tag))

    return answer


def cut_no_tag(path):
    jieba.load_userdict('pku_training_words_dict.utf8')

    lines = codecs.open(path, 'r', 'utf-8')
    answer=[]
    for line in lines:
        line = line.replace(' ', '')
        cut=jieba.cut(line)
        answer.append(' '.join(cut))
    return answer


def write_sen(line_list,path):
    all_string=''
    for line in line_list:
        all_string+=line
    write_file=codecs.open('write_file'+path+'_no_tag','w','utf-8')
    write_file.write(all_string)


def prepare_sentence(path,cut_and_tag=True):
    lines = codecs.open(path, 'r', 'utf-8')
    all_list=[]
    if cut_and_tag is True:
        for line in lines:
            word_list=[]
            tag_list=[]
            line =line.replace(' ','').rstrip()
            word_flag=pseg.cut(line)
            for word,tag in word_flag:
                word_list.append(word)
                tag_list.append(tag)
            assert len(word_list)==len(tag_list)
            all_list.append((word_list,tag_list))
    else:
        for line in lines:

            words_and_tags=line.split(' ')
            words=[word_tag.split('/')[0]   for word_tag in words_and_tags if word_tag!='\n']
            tags=[word_tag.split('/')[1]  for word_tag in words_and_tags if word_tag!='\n']
            all_list.append((words,tags))

    assert all_list is not None
    return all_list    #[[([],[])],[],...    ]


def prepare_test_sen(path):
    lines=codecs.open(path,'r','utf-8')
    all_list=[]
    for line in lines:
        words=line.rstrip().split('  ')
        assert words is not None
        all_list.append(words)
    assert all_list is not None
    return all_list#每一个元素是一个list，每个list是一句话，包含n个词


def get_tag_dict(path):
    tag_dict={}
    tag_name_dict={}
    lines=codecs.open(path,'r','utf-8')

    for ix,line in enumerate(lines):
        tag_v=line.replace('\n','').split(' ')
        assert len(tag_v)==2
        tag_dict[tag_v[0]]=ix
        tag_name_dict[tag_v[0]]=tag_v[1]
    tag_dict['<START>']=len(tag_dict)
    tag_dict['<STOP>']=len(tag_dict)
    tag_name_dict['<START>']='开始'
    tag_name_dict['<STOP>']='结束'
    return tag_dict,tag_name_dict


def print_answer(tuple,ix_to_tag,ix_to_word):
    words_list=tuple[2].detach().numpy()
    tags_list=tuple[1]
    tags=[ix_to_tag[i] for i in tags_list]
    words=[ix_to_word[i] for i in words_list]
    assert len(tags)==len(words)
    for w,t in zip(words,tags):
        print (w,t)


def saving_answer_pre_sen(tuple,ix_to_tag,ix_to_word,true_sen):#对于每一行
    #words_list = tuple[2].detach().numpy()#得到的是一个二维tensor
    tags_list = tuple[1]
    tags = [ix_to_tag[i] for i in tags_list]
    #得到单词
    #words=[]
    #for row in words_list:
    #    for (key,v) in ix_to_word.items():
    #        if (row == v.detach().numpy()).all==True:
    #            words.append(key)
    #            break
    #    print 'tensor to word error'


    #words = [ix_to_word[i] for i in words_list]
    #assert len(tags)==len(words)
    t=(true_sen,tags)



    return t #每一个元素是一个数对，单词和标签

if __name__=='__main__':
    lines= codecs.open('train_plants.txt','r','utf-8')
    result=[]
    for ix, line in enumerate(lines):
        if ix>40004:
            break
        else:
           result.append(str(line))
    wr=codecs.open('train_40000_lines.txt','w','utf-8')
    for string in result:
        wr.write(string )