# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 20:05:22 2014

@author: nunomarques
"""
"""
#RUN ONCE
train=pd.read_csv("./data/train.csv")
test=pd.read_csv("./data/test.csv")
"""
"""
import nn
reload(nn)
import nn_mnist
reload(nn_mnist)
from nn_mnist import *
from nn import *
nn=NNet_test()
nn.set_dataset(train)
nn.set_alpha(0.5)
nn.set_n_iter(3000)
nn.set_lambd(0)
nn.set_batch_size(100)
nn.set_epsilon(0.99)
nn.set_weight_update("momentum")

nn.add_n_layers(4)

nn.get_input_layer().add_n_vertex(1)
nn.get_input_layer().get_vertex(0).set_num_nodes(784)

nn.get_layer(1).add_n_vertex(1)
nn.get_layer(1).get_vertex(0).set_num_nodes(784)
nn.get_layer(1).get_vertex(0).add_in_edge(nn.get_input_layer().get_vertex(0))

nn.get_layer(2).add_n_vertex(1)
nn.get_layer(2).get_vertex(0).set_num_nodes(300)
nn.get_layer(2).get_vertex(0).add_in_edge(nn.get_layer(1).get_vertex(0))

nn.get_output_layer().add_n_vertex(1)
nn.get_output_layer().get_vertex(0).set_num_nodes(10)
nn.get_output_layer().get_vertex(0).add_in_edge(nn.get_layer(2).get_vertex(0))

nn.initialize_data()
nn.initialize_weight_matrixes(0.01)
nn.initialize_deltas()
#nn.set_data()
nn.grad_descent()
p=nn.predict(test)
"""

from nn import *
import scipy.io
class NNet_test(NNet):
    
    def set_data(self):
        idxs=np.random.randint(self.dataset.shape[0],size=self.batch_size)
        self.get_input_layer().get_vertex(0).set_data(self.dataset[idxs,1:])
        self.set_target_variable(self.dataset[idxs,0:1],np.unique(self.dataset[:,0:1]))
        
    def set_dataset(self,d):
        self.dataset=d.as_matrix()
        
    def gradient_check(self,epsilon):
        self.initialize_gradient_checking()
        for layer_idx in range(self.num_layers()-1):
            for vertex in self.get_layer(layer_idx).get_vertexes():
                for i in range(vertex.weight_matrix.shape[0]):
                    for j in range(vertex.weight_matrix.shape[1]):
                        vertex.weight_matrix[i][j]+=epsilon
                        self.forward_prop()
                        self.compute_cost()
                        cost_plus=self.get_cost()
                        vertex.weight_matrix[i][j]-=(2*epsilon)
                        self.forward_prop()
                        self.compute_cost()
                        cost_minus=self.get_cost()
                        vertex.weight_matrix[i][j]+=epsilon
                        vertex.grad_check[i][j]=(cost_plus-cost_minus)/(2*epsilon)
                        
    def initialize_gradient_checking(self):
        for l_idx in range(self.num_layers()-1):
            for vertex in self.get_layer(l_idx).get_vertexes():
                vertex.grad_check=0*vertex.weight_matrix
    
    def set_target_variable(self,y,classes):
        self.target_variable=extend_target_variable(y,classes)
        
    def predict(self,test):
        self.set_batch_size(test.shape[0])
        self.initialize_data()
        self.get_input_layer().get_vertex(0).set_data(test.as_matrix())
        self.forward_prop()
        return self.get_estimate()
    
    