# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 01:13:08 2020
Edited on Fri Dec 16 15:28:00 2022

@author: LW Kong
"""
# with tensorflow 2
# Adaption
# W required or not
# using phi function

# 4_1
# need to track nan

# features to be added
# what is the source of Nan?
# set ranges for the variables
# give up two-level train functions, use if mod() to plot


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import time
tic = time.time()

tick_plot_size = 32
label_plot_size = 32


####################################################################
##Parameters
input_0 = 0.06 # input value as the 0 state
input_1 = 0.6 # input value as the 1 state
# the choices of 0.06 and 0.6 come from Wenjia Shi, ... Chao Tang 2017 Sci Rep

dt 		= 0.02		# length of each time step

T 		= 60*50		# number of time steps
T_switch = 30*50    # the time step to switch the input signal (from input_0 to input_1)
T_c     =  2*50
T_peak  =  8*50
T_plot_cut = 10*50  # cut the initial part from plotting



B 		= 1    	# batch size (size of the ensemble of inputs) 

####################################################################
## Network Topology
#         AA,AB,AC,  BA,BB,BC,  CA,CB,CC
#network = [0,1,1,     0,0,-1,    0,0,0];
network = [0,-1,1,     0,0,1,    0,0,0];
#         AA,BA,CA   AB,BB,CB   AC,BC,CC
#networkT = [0,0,0,     1,0,0,    1,-1,0]

network_i = 20
network_data = pd.read_csv('nfb.txt', sep='\t')
# choose network topology from a given family


network_df_temp = network_data.iloc[network_i,1:10]
#network = network_df_temp.values.tolist()

N = np.sqrt(len(network)) # number of nodes
N = N.astype(int)

L = np.sum(np.absolute(network)) # number of links
L = L.astype(int)
L = L + 1

# transpose
networkT = []
for n_i in range(N):
    for n_j in range(N):
        networkT.append( network[ n_i+N*n_j ] )

####################################################################
## Variables to be optimized (gene circuit parameters)
produce_N = tf.Variable( tf.zeros([N],dtype = tf.float64) )
decay_N = tf.Variable( tf.zeros([N],dtype = tf.float64) )

pow_L = tf.Variable( tf.zeros([L],dtype = tf.float64) )
kp_L = tf.Variable( tf.zeros([L],dtype = tf.float64) )

## Other variables
NetworkState = tf.Variable( tf.zeros([B,N],dtype = tf.float64) )
NetworkOutput = tf.Variable( tf.zeros([B],dtype = tf.float64) )

inputVal = tf.Variable( tf.zeros([T, B, N],dtype = tf.float64) )
initialVal = tf.Variable( tf.zeros([B, N],dtype = tf.float64) )

####################################################################
## Initialize weights (gene circuit parameters)
def init_weights():
    produce_N.assign(tf.random.uniform([N],minval=0.0, maxval=1.0,dtype = tf.float64))
    decay_N.assign(tf.random.uniform([N],minval=0.0, maxval=1.0,dtype = tf.float64))
    pow_L.assign(tf.random.uniform([L],minval=1.0, maxval=4.0,dtype = tf.float64))
    kp_L.assign(tf.math.pow( 10.0, tf.random.uniform([L],minval=-2.0, maxval=1.0,dtype = tf.float64) ))

####################################################################
##ODE function


# Hill equation for regulatory networks with an arbitary size
@tf.function
def fn_links(s,i):
    
    # to A
    l_i = 0
    np = tf.slice(pow_L,[l_i],[1])
    kp = tf.slice(kp_L,[l_i],[1])
    x  = tf.slice(i,[0,0],[1,1])
    da = tf.math.divide( tf.math.pow(x,np), tf.math.pow(x,np)+ kp )
    
    for n_i in range(N):
        if networkT[n_i] == 1:
            l_i = l_i + 1
            np  = tf.slice(pow_L,[l_i],[1])
            kp  = tf.slice(kp_L,[l_i],[1])
            x   = tf.slice(s,[0,n_i],[1,1])
            da  = da * tf.math.divide( tf.math.pow(x,np), tf.math.pow(x,np)+ kp )
        elif networkT[n_i] == -1:
            l_i = l_i + 1
            np  = tf.slice(pow_L,[l_i],[1])
            kp  = tf.slice(kp_L,[l_i],[1])
            x   = tf.slice(s,[0,n_i],[1,1])
            da  = da * tf.math.divide( kp, tf.math.pow(x,np)+ kp )
    
    # to B
    if networkT[N] != 0:
        start_i = 0
    elif networkT[N+1] != 0:
        start_i = 1
    else:
        start_i = 2
    
    l_i = l_i+1
    np = tf.slice(pow_L,[l_i],[1])
    kp = tf.slice(kp_L,[l_i],[1])
    x  = tf.slice(s,[0,start_i],[1,1])
    if networkT[N+start_i] == 1:
        db = tf.math.divide( tf.math.pow(x,np), tf.math.pow(x,np)+ kp )
    else:
        db = tf.math.divide( kp, tf.math.pow(x,np)+ kp )
    
    for n_i in range(start_i+1,N):
        if networkT[N+n_i] == 1:
            l_i = l_i + 1
            np  = tf.slice(pow_L,[l_i],[1])
            kp  = tf.slice(kp_L,[l_i],[1])
            x   = tf.slice(s,[0,n_i],[1,1])
            db  = db * tf.math.divide( tf.math.pow(x,np), tf.math.pow(x,np)+ kp )
        elif networkT[N+n_i] == -1:
            l_i = l_i + 1
            np  = tf.slice(pow_L,[l_i],[1])
            kp  = tf.slice(kp_L,[l_i],[1])
            x   = tf.slice(s,[0,n_i],[1,1])
            db  = db * tf.math.divide( kp, tf.math.pow(x,np)+ kp )
    
    # to C
    if networkT[N+N] != 0:
        start_i = 0
    elif networkT[N+N+1] != 0:
        start_i = 1
    else:
        start_i = 2
    
    l_i = l_i+1
    np = tf.slice(pow_L,[l_i],[1])
    kp = tf.slice(kp_L,[l_i],[1])
    x  = tf.slice(s,[0,start_i],[1,1])
    if networkT[N+N+start_i] == 1:
        dc = tf.math.divide( tf.math.pow(x,np), tf.math.pow(x,np)+ kp )
    else:
        dc = tf.math.divide( kp, tf.math.pow(x,np)+ kp )
    
    for n_i in range(start_i+1,N):
        if networkT[N+N+n_i] == 1:
            l_i = l_i + 1
            np  = tf.slice(pow_L,[l_i],[1])
            kp  = tf.slice(kp_L,[l_i],[1])
            x   = tf.slice(s,[0,n_i],[1,1])
            dc  = dc * tf.math.divide( tf.math.pow(x,np), tf.math.pow(x,np)+ kp )
        elif networkT[N+N+n_i] == -1:
            l_i = l_i + 1
            np  = tf.slice(pow_L,[l_i],[1])
            kp  = tf.slice(kp_L,[l_i],[1])
            x   = tf.slice(s,[0,n_i],[1,1])
            dc  = dc * tf.math.divide( kp, tf.math.pow(x,np)+ kp )
    
        
    
    Hill_output = tf.stack([da,db,dc]) #shape(3,1,1)
    Hill_output = tf.reshape(Hill_output,[1,3])    
    return Hill_output

@tf.function
def Output():
    output = tf.scan(lambda s,i: s + dt*(produce_N*fn_links(s,i)-decay_N*s),
							elems = inputVal,
                            initializer = initialVal,
                            swap_memory = True) # True enables GPU-CPU memory swapping
    output = tf.slice(output,[0,0,N-1],[T,B,1])
    output = tf.reshape(output,[T,B])
    return output
####################################################################
## Loss Function
@tf.function
def Loss():
    O = Output() # [T,B]
    Opeak = tf.reduce_max(tf.slice(O,[T_switch,0],[T_peak,B]),0)
    O1 = tf.reduce_mean(tf.slice(O,[T_switch-T_c,0],[T_c,B]),0)
    O2 = tf.reduce_mean(tf.slice(O,[T-T_c,0],[T_c,B]),0)
    
    #return ( - tf.math.log( tf.reduce_mean( Opeak-O1) ) + tf.math.log( tf.reduce_mean( tf.abs(O2-O1) ) ) )
    return ( - tf.math.sqrt( tf.reduce_mean( tf.abs(Opeak-O1)) ) + tf.math.sqrt( tf.reduce_mean( tf.abs(O2-O1) ) ) )
    
    #Sensi = tf.reduce_mean( tf.divide( Opeak-O1, O1 ))
    #Preci = tf.reduce_mean( tf.abs( tf.divide(O1, O2-O1) ))
    #return ( - tf.math.sqrt(Sensi) - tf.math.sqrt(Preci) )

## return sensitivity and precision
def SP():
    O = Output() # [T,B]
    Opeak = tf.reduce_max(tf.slice(O,[T_switch,0],[T_peak,B]),0)
    O1 = tf.reduce_mean(tf.slice(O,[T_switch-T_c,0],[T_c,B]),0)
    O2 = tf.reduce_mean(tf.slice(O,[T-T_c,0],[T_c,B]),0)
    #SensiVal.assign(tf.math.sqrt( tf.reduce_mean( Opeak-O1) ))
    #PreciVal.assign(tf.reduce_mean( tf.divide( O1, O2-O1 )))
    #PreciVal.assign(tf.math.sqrt( tf.reduce_mean( tf.abs(O2-O1) ) ) )
    
    # classic
    SensiVal = tf.reduce_mean( tf.divide( tf.abs( Opeak-O1 ), O1 ))
    PreciVal = tf.reduce_mean( tf.math.log( tf.divide( O1, tf.abs(O2-O1 ))))
    return SensiVal, PreciVal
####################################################################
## Optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.002)
####################################################################
## Initialize the Circuit: Prepare initial values and input time series
def initCircuit(): 
    # assign initialVal
    initialVal.assign( 0.1*np.ones([B,N]) ) 
    # assign inputVal
    inputNoNoise = input_0*np.ones(B)
    input_temp = inputNoNoise.reshape(B,1) * np.random.normal(loc=1.0, scale = 0.01, size=[T,B,N])
    input_temp[T_switch:T,:,0] = input_temp[T_switch:T,:,0] + input_1
    input_temp[:,:,1:N] = 0.0    
    inputVal.assign(input_temp)

####################################################################
## Training model function
def trainIter(LossHistory,SensiHistory,PreciHistory,iterations=1,plotLossVal=False,plotLoss=True):        
    for i in range(iterations):
        initCircuit()        
        opt.minimize(Loss,[produce_N,decay_N,pow_L,kp_L])
        l = Loss()
        SensiVal, PreciVal = SP()    
        LossHistory = np.append(LossHistory,l.numpy())
        SensiHistory = np.append(SensiHistory,SensiVal.numpy())
        PreciHistory = np.append(PreciHistory,PreciVal.numpy())
        if(plotLoss):
            print('Loss: ',l.numpy())
        if(plotLossVal):
            print('Sensi: ',SensiVal.numpy(),' Prec: ',PreciVal.numpy())
        if(np.isnan(l)):
            b_NoNan = False
            return LossHistory,SensiHistory,PreciHistory,b_NoNan
    b_NoNan = True
    return LossHistory,SensiHistory,PreciHistory,b_NoNan

# simulate the circuit without optimizing, and plot
def simulateModel():
    initCircuit()

    finalInput = inputVal.numpy()
    finalInput = finalInput[:,0,0]
    
    finalOutput = Output().numpy()
    finalOutput = finalOutput[:,0]
    finalOutput = np.reshape(finalOutput,T)
    t_plot = np.linspace(0,T*dt,T)
    #plt.plot(t_plot,2.0*finalOutput,'r--',t_plot,finalInput,'b')
    plt.figure(figsize=(15,9))
    plt.plot(t_plot[T_plot_cut:T],finalOutput[T_plot_cut:T],'r--')
    #print(finalOutput.shape)
    plt.plot([T_switch*dt,T_switch*dt],[np.min(finalOutput),np.max(finalOutput) ],'b--')
    plt.xlabel('time', fontsize=label_plot_size)
    plt.ylabel('response', fontsize=label_plot_size)
    plt.rc('xtick', labelsize=tick_plot_size) 
    plt.rc('ytick', labelsize=tick_plot_size) 
    plt.show()

def train(nn_iter,n_iter):
    LossHistory = np.zeros([])
    SensiHistory = np.zeros([])
    PreciHistory = np.zeros([])
           
    b_NoNan = True
    for i in range(nn_iter):
        if b_NoNan == True:
            LossHistory,SensiHistory,PreciHistory,b_NoNan = \
                trainIter( LossHistory,SensiHistory,PreciHistory,iterations=n_iter)
            simulateModel() # simulate and plot
            
    # at the end, plot the histories of performance over iterations
    if b_NoNan == True:
        simulateModel()
        
        LossHistory = LossHistory[1:np.size(LossHistory)]
        plt.figure(figsize=(15,9))
        plt.plot(LossHistory)
        plt.xlabel('epoch', fontsize=label_plot_size)
        plt.ylabel('Loss', fontsize=label_plot_size)
        plt.rc('xtick', labelsize=tick_plot_size) 
        plt.rc('ytick', labelsize=tick_plot_size) 
        plt.show()
        
        SensiHistory = SensiHistory[1:np.size(SensiHistory)]
        plt.figure(figsize=(15,9))
        plt.plot(SensiHistory)
        plt.xlabel('epoch', fontsize=label_plot_size)
        plt.ylabel('Sensitivity', fontsize=label_plot_size)
        plt.rc('xtick', labelsize=tick_plot_size) 
        plt.rc('ytick', labelsize=tick_plot_size) 
        plt.show()
        
        PreciHistory = PreciHistory[1:np.size(PreciHistory)]
        plt.figure(figsize=(15,9))
        plt.plot(PreciHistory)
        plt.xlabel('epoch', fontsize=label_plot_size)
        plt.ylabel('ln Precision', fontsize=label_plot_size)
        plt.rc('xtick', labelsize=tick_plot_size) 
        plt.rc('ytick', labelsize=tick_plot_size) 
        plt.show()
    
        print('k_node')
        k_node = np.transpose(np.stack( (produce_N.numpy(), decay_N.numpy()) ))
        print(k_node)
        print('k_link')
        k_link = np.transpose(np.stack( (pow_L.numpy(),kp_L.numpy()) ))
        print(k_link)
    return b_NoNan
####################################################################
## Train model
nn_iter = 4 # num of larger loop, also the number of generated plots
n_iter = 5 # num of smaller loop, also plot every n_iter iterations

init_weights()
b_NoNan = train(nn_iter,n_iter)



toc = time.time() - tic
print('running time ' + str(toc) + 's')


# 50 min for 7 i_NoNan, with 250 iters
# 8 min each




