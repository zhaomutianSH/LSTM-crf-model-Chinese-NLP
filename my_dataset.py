#coding=utf-8
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


def index_to_one_hot(ix,total_number):
    a=torch.zeros([total_number])
    a[ix]=1
    return a


def coll_fn(batch):
    return batch


class my_data_set(Data.Dataset):
    def __init__(self,tuple_list,if_cuda):
        self.data=[]
        if if_cuda:
            for tup in tuple_list:
                if(type(tup[0])==torch.Tensor):
                    a=tup[0].cuda()
                else:
                    a=tup[0]
                if (type(tup[1]) == torch.Tensor):
                    b=tup[1].cuda()
                elif(type(tup[1])==list):
                    b=[i.cuda() for i in tup[1]]
                else:
                    print('input data format error')
                self.data.append((a,b))
            print('gpu')
        else:
            self.data=tuple_list
            print('cpu')



    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def getitem(self, item):
        return self.data[item]


class Embeding_data_set():
    def __init__(self,list_of_tuple,words_number,if_cuda):
        self.data=list_of_tuple
        self.if_cuda=if_cuda
        self.num=words_number
    def __len__(self):
        return len(self.data)

    def getitem(self, item):

        lable=self.data[item][0]
        con=[]
        for i in self.data[item][1]:
            if self.if_cuda:
                con.append(index_to_one_hot(i,self.num).cuda())
            else:
                con.append(index_to_one_hot(i, self.num))

        return lable,con
