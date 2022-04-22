# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 14:48:06 2022

@author: yuyi6
"""

from delicate.kd.kd_model_class import MolBertconfig, MolBertForDistill
from delicate.clps_kd.delicate_model import PSMolbertdistill,PSMolBertconfig
import datetime
import torch

def speed_comparison(student):
    
    config_T = MolBertconfig(num_hidden_layers=12)
    model_T = MolBertForDistill(config_T)
    
    if student == 'KDMolBERT':
        
        config_S = MolBertconfig(num_hidden_layers=3)
        model_S = MolBertForDistill(config_S)
        
    elif student == 'PSMolBERT':
        
        config_S = PSMolBertconfig(num_hidden_layers=12)
        model_S = PSMolbertdistill(config_S)
        
    elif student == 'DeLiCate':
        config_S = PSMolBertconfig(num_hidden_layers=3)
        model_S = PSMolbertdistill(config_S)
    
    else:
        print('there is no such model')
    
    device = torch.device('cuda')
    model_T.to(device)
    model_S.to(device)
    
    
    torch.cuda.empty_cache()
    dummy_input = torch.zeros(32,128,dtype = torch.int64).to(device)
    repetitions = 10000
    
    #warm up GPU
    for _ in range(10):
       aa,_,_,_,_ = model_T(dummy_input,dummy_input,dummy_input,dummy_input)
       
    starttime = datetime.datetime.now()
    
    with torch.no_grad():
      for rep in range(repetitions):
          aa,_,_,_,_ = model_S(dummy_input,dummy_input,dummy_input,dummy_input)
    endtime = datetime.datetime.now()
    time_1 = endtime.timestamp()-starttime.timestamp()
    
    starttime = datetime.datetime.now()
    with torch.no_grad():
      for rep in range(repetitions):
          aa,_,_,_,_ = model_T(dummy_input,dummy_input,dummy_input,dummy_input)
    endtime = datetime.datetime.now()
    time_2 = endtime.timestamp()-starttime.timestamp()
    
    print('Speedup:',str(round(time_2/time_1,2)),'times')