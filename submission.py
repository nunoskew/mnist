# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 14:07:28 2014

@author: skew
"""
from pre_process import *
from nn_mnist import *   
import numpy as np
import pandas as pd

def submit(filename):
    """(string) -> None
    
    Runs perceptron on the titanic dataset and generates
    a submission file 'filename'.csv
    
    """
    train=pd.read_csv("./data/train.csv")
    test=pd.read_csv("./data/test.csv")
    nn=NNet_test()
    nn.set_dataset(train)
    nn.set_alpha(0.1)
    nn.set_n_iter(3000)
    nn.set_lambd(0)
    nn.set_batch_size(1000)
    
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
    nn.initialize_weight_matrixes(0.12)
    nn.initialize_deltas()
    #nn.set_data()
    nn.grad_descent()    
    #read data
    p=nn.predict(test)
    #merge PassengerId and predictions to create the submission DataFrame
    df=pd.DataFrame(np.concatenate([np.array(range(1,28001)).reshape(28000,1),\
                    np.argmax(p,axis=1).reshape(28000,1)],axis=1))
    #set up column names
    df.columns=['ImageId','Label']
    #create submission csv
    df.to_csv('./submissions/'+filename,header=True,index=False)