# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 17:33:00 2014

@author: skew
"""
import pandas as pd
import numpy as np
from nn import *
"""
import nn
reload(nn)
from nn import *
#train,_=read_data()
nn=NNet()
nn.dataset=train
nn.set_alpha(0.01)
nn.set_n_iter(5000)
nn.set_lambd(0)
nn.set_batch_size(100)

nn.add_n_layers(3)

nn.get_input_layer().add_n_vertex(1)
nn.get_input_layer().get_vertex(0).set_num_nodes(784)

nn.get_layer(1).add_n_vertex(1)
nn.get_layer(1).get_vertex(0).set_num_nodes(784)
nn.get_layer(1).get_vertex(0).add_in_edge(nn.get_input_layer().get_vertex(0))

nn.get_output_layer().add_n_vertex(1)
nn.get_output_layer().get_vertex(0).set_num_nodes(10)
nn.get_output_layer().get_vertex(0).add_in_edge(nn.get_layer(1).get_vertex(0))

nn.initialize_data()
nn.initialize_weight_matrixes(0.01)
nn.initialize_deltas()

nn.grad_descent()
"""
def read_data():
    train=pd.read_csv("./data/train.csv")
    test=pd.read_csv("./data/test.csv")
    
def set_data(nnet):
        #m=112
        #idxs=random.sample(xrange(m),m-self.batch_size)
        idxs=np.random.randint(nnet.dataset.shape[0],size=nnet.batch_size)
        #dataset=pd.read_table("ok.txt",",",header=0,skiprows=idxs)
        #relation=pd.get_dummies(dataset['relation']).as_matrix()
        #person1=pd.get_dummies(dataset['person1']).as_matrix()
        #person2=pd.get_dummies(dataset['person2']).as_matrix()
        nnet.get_input_layer().get_vertex(0).set_data(nnet.dataset[idxs,1:])
        nnet.set_target_variable(nnet.dataset[idxs,0])
        
def set_dataset(nnet,d):
        nnet.dataset=data