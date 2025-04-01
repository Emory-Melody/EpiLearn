
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import time
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import json
# import argparse
# import random 
# from torch.nn.utils.rnn import pad_sequence
# dtype = torch.float
# import itertools
# from sklearn.preprocessing import StandardScaler
# from model_utils import *
# from data_utils import * 
# import pdb 

# dtype = torch.float
# regions = all_hhs_regions  # for rnn


# class EINN(nn.Module):
#     def __init__(
#         self,
#         args: argparse.ArgumentParser,
#         ):
#         super(EINN, self).__init__()

#         torch.manual_seed(SEED)
#         random.seed(SEED)
#         np.random.seed(SEED)

#         # read testbed setup
#         self.ode_model = 'SEIRm'  
#         self.exp = args.exp
#         self.get_hyperparameters()
#         self.pred_week = args.pred_week
#         self.region = args.region
#         self.device = torch.device(args.dev)
#         self.model_name = 'EINN' + '-' + self.ode_model
#         self.start_week = EW_START_DATA
#         min_sequence_length = 20

#         start = time.time()

#         ''' Process user input regarding regions '''
#         global all_hhs_regions, regions
#         if self.region=='all':
#             regions = all_hhs_regions
#         else:
#             regions = self.region
#         if type(regions) is str:
#             regions = list([regions])

#         ''' Import data for all regions '''
#         self.initial_conditions = {}; self.end_point = {}
#         r_seqs = []  # state sequences of features
#         r_ys = []  # state targets
#         r_S_rmse = []  # state rmse
#         for region in regions:
#             X_state, y = get_state_train_data(region,self.pred_week)
#             r_seqs.append(X_state.to_numpy())
#             r_ys.append(y)
#             ''' Import RMSE calibration data '''
#             try:
#                 S_rmse = load_rmse_data(self.start_week,self.pred_week,region)
#                 # change from M to m (from cumulative to incidence mortality)
#                 S_rmse[:,4] = y.ravel()
#             except:
#                 print(region)  
#                 raise Exception(f'not {region} in ODE calibration')  
#             self.initial_conditions[region] = S_rmse[0,:]
#             self.end_point[region] = S_rmse[-1,:]
#             r_S_rmse.append(S_rmse)
#         r_seqs = np.array(r_seqs)  # shape: [regions, time, features]
#         r_ys = np.array(r_ys)  # shape: [regions, time, 1]
#         r_S_rmse = np.array(r_S_rmse)  # shape: [regions, time, 1]

#         # Normalize
#         # One scaler per state
#         seq_scalers = dict(zip(regions, [StandardScaler() for _ in range(len(regions))]))
#         ys_scalers = dict(zip(regions, [TorchStandardScaler() for _ in range(len(regions))]))
#         rmse_scalers = dict(zip(regions, [TorchStandardScaler() for _ in range(len(regions))]))
#         r_seqs_norm = []; r_ys_norm = []; r_rmse_norm = []
#         self.S_scale = {} # dictionary of scaling vectors for each region
#         for i, r in enumerate(regions):
#             r_seqs_norm.append(seq_scalers[r].fit_transform(r_seqs[i],self.device))
#             r_ys_norm.append(ys_scalers[r].fit_transform(r_ys[i],self.device))
#             r_rmse_norm.append(rmse_scalers[r].fit_transform(r_S_rmse[i],self.device))
#             # save rmse std
#             self.S_scale[r] = rmse_scalers[r].std.reshape(-1)
#         r_seqs_norm = np.array(r_seqs_norm)
#         r_ys_norm = np.array(r_ys_norm)
#         r_rmse_norm = np.array(r_rmse_norm)
#         # two of them are used during training
#         self.rmse_scalers = rmse_scalers
#         self.ys_scalers = ys_scalers

#         ''' Prepare train and validation dataset '''

#         def create_time_seq(no_sequences, sequence_length):
#             """
#                 Creates windows of fixed size
#             """
#             # convert to small sequences for training, starting with length 10
#             seqs = []
#             # starts at sequence_length and goes until the end
#             for idx in range(no_sequences):
#                 # Sequences
#                 seqs.append(torch.arange(idx,idx+sequence_length))
#             seqs = pad_sequence(seqs,batch_first=True).type(dtype)
#             return seqs
        
#         states, seqs, seqs_masks, y, y_mask, y_weights, rmse_seqs, time_seqs = [], [], [], [], [], [], [], []
#         test_states, test_seqs, test_seqs_masks, test_time_seq = [], [], [], []
#         for region, seq, ys, rmse in zip(regions, r_seqs_norm, r_ys_norm, r_rmse_norm):
#             ys_weights = np.ones((ys.shape[0],1))
#             ys_weights[-14:] *= 5 
#             seq, seq_mask, ys, ys_mask, ys_weight, rmse_seq = create_window_seqs(seq,rmse,ys,ys_weights,min_sequence_length)
#             # normal
#             states.extend([region for _ in range(seq.shape[0])])
#             seqs.append(seq)
#             seqs_masks.append(seq_mask)
#             y.append(ys)
#             y_mask.append(ys_mask)
#             # ys weights
#             y_weights.append(ys_weight.squeeze_(2))
#             rmse_seqs.append(rmse_seq)
#             # time sequences
#             time_seq = create_time_seq(seq.shape[0],min_sequence_length+WEEKS_AHEAD*DAY_WEEK_MULTIPLIER).unsqueeze(2)
#             time_seqs.append(time_seq)
#             # now fill up the test data
#             test_states.append(region)
#             test_seqs.append(seq[[-1]]); test_seqs_masks.append(seq_mask[[-1]])
#             test_time_seq.append(time_seq[[-1]])

#         # train and validation data, combine 
#         regions_train = np.array(states, dtype="str").tolist()
#         X_train = torch.cat(seqs,axis=0).float().numpy()
#         X_mask_train = torch.cat(seqs_masks,axis=0).unsqueeze(2).float().numpy()
#         y_train = torch.cat(y,axis=0).float().numpy()
#         y_mask_train = torch.cat(y_mask,axis=0).float().numpy()
#         y_weights_train = torch.cat(y_weights,axis=0).float().numpy()
#         rmse_train = torch.cat(rmse_seqs,axis=0).float().numpy()
#         time_train = torch.cat(time_seqs,axis=0).float().numpy()


#         # same for test
#         regions_test = np.array(test_states, dtype="str").tolist() 
#         X_test = torch.cat(test_seqs,axis=0).float().numpy()
#         X_mask_test = torch.cat(test_seqs_masks,axis=0).unsqueeze(2).float().numpy()
#         time_test = torch.cat(test_time_seq,axis=0).float().numpy()
#         # for scaling time module
#         self.t_min = torch.tensor(time_train.min())
#         # t_max is also useful for ode future 
#         self.t_max = torch.tensor(time_train.max())

#         # convert dataset to use in dataloader
#         train_dataset = SeqData(regions_train, X_train, X_mask_train, y_train, y_mask_train, y_weights_train, rmse_train, time_train)
#         # note: y_val, y_mask_val, y_weights_val, rmse_val, are not needed at test time
#         empty = np.zeros_like(regions_test)
#         test_dataset = SeqData(regions_test, X_test, X_mask_test, empty, empty, empty, empty, time_test)
        
#         # create dataloaders for each region
#         self.data_loaders = {}
#         for r in regions:
#             idx = torch.tensor(np.isin(train_dataset.region,r)) #== r
#             dset_train = torch.utils.data.dataset.Subset(train_dataset, np.where(idx)[0])
#             r_train_loader = torch.utils.data.DataLoader(dset_train, batch_size=self.batch_size, shuffle=True, pin_memory=True) 
#             self.data_loaders[r] = r_train_loader
#         print(time.time() - start ,' seconds')
#         # test data loader is small so we can use only one
#         self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

#         """ create set of ODEs""" 
#         df = pd.read_csv(population_path,index_col='region')
#         self.pop = {}
#         for region in regions:
#             self.pop[region] = df.at[region,'PopulationEstimate2019']

#         """ Create models, one per region """
#         self.encoder = EncoderModules(regions,X_train.shape[2],self.device)
#         self.decoder = DecoderModules(regions,self.device)
#         self.out_layer = OutputModules(regions,self.device)
#         out_layer_width = 20
#         self.time_nn_mod = \
#             time_pnn_fourier(
#                 regions=regions,
#                 scale=1,
#                 out_dim=out_layer_width,
#                 device=self.device
#             ).to(self.device)

#         self.ode = \
#             ode_modules_set(\
#                 regions=regions,
#                 init_ode_wcalibration=self.init_ode_wcalibration,
#                 initial_ode_idx=0,
#                 final_ode_idx=N_ODES_MAX,
#                 device=self.device,
#                 pop=self.pop
#             ).to(self.device)
                
#         """ paths to save model """
#         self.mod_path = './models/EINN/exp{}/{}/'.format(self.exp,self.region)
#         self.rnnode_file_name = 'einn_{}_{}'.format(self.pred_week,self.ode_model)

#         self.losses = []
#         self.ode_losses = []
#         self.data_losses = []
#         self.aux_losses = []
#         self.feat_losses = []
#         self.future_ode_losses = []
#         self.kd_target_losses = []; self.kd_emb_losses = []; self.feat_data_losses = []
#         self.gradient_losses = []
#         self.ode_param_losses = []
#         self.monotonicity_losses = []
#         self.lr_values = []; self.lr_ode_values = []; self.lr_values = []
#         self.epoch = 0
#         self.best_loss = torch.tensor(np.inf,dtype=dtype).to(self.device)
#         self.early = False
#         self.train_start_flag = []
#         self.save_load_ode_params = True
#         self.feat_mode = True


#     def save_odernn_local(self,suffix):

#         print('==== saving model ====')
#         def save_model(file_prefix: str, model: nn.Module, region: str):
#             torch.save(model.encoder.mods['encoder_'+region].state_dict(), file_prefix + "_encoder.pth")
#             torch.save(model.decoder.mods['decoder_'+region].state_dict(), file_prefix + "_decoder.pth")
#             torch.save(model.time_nn_mod.time_mods['time_nn_'+region].state_dict(), file_prefix + "_time_nn.pth")
#             if self.save_load_ode_params:
#                 torch.save(model.ode.state_dict(), file_prefix + "_ode.pth")
#             torch.save(model.out_layer.mods['output_'+region].state_dict(), file_prefix + "_out_layer.pth")

#         file_name = 'einn_{}_{}{}'.format(self.pred_week,self.ode_model,suffix)
#         for region in regions:
#             mod_path = './models/EINN/exp{}/{}/'.format(self.exp,region)
#             if not os.path.exists(mod_path):
#                 os.makedirs(mod_path)
#             save_model(mod_path+file_name,self,region)    

#         print('==== saved ====')    
        

#     def upload_odernn_local(self,suffix=''):

#         ''' this works well, saves regional model in the correct one'''

#         def load_model(file_prefix: str, model: nn.Module, region: str):
#             model.encoder.mods['encoder_'+region].out_layer[0].bias
#             model.encoder.mods['encoder_'+region].load_state_dict(torch.load(file_prefix + "_encoder.pth",map_location=self.device),strict=False)
#             model.decoder.mods['decoder_'+region].load_state_dict(torch.load(file_prefix + "_decoder.pth",map_location=self.device),strict=False)
#             model.time_nn_mod.load_state_dict(torch.load(file_prefix + "_time_nn.pth",map_location=self.device),strict=False)
#             if self.save_load_ode_params:
#                 model.ode.load_state_dict(torch.load(file_prefix + "_ode.pth",map_location=self.device),strict=False)
#             model.out_layer.mods['output_'+region].load_state_dict(torch.load(file_prefix + "_out_layer.pth",map_location=self.device),strict=False)

#         if self.pretrained == 'None':
#             experiment = self.exp
#         else:
#             experiment = self.pretrained

#         print('\n==== loading model ====')
#         file_name = 'einn_{}_{}{}'.format(self.pred_week,self.ode_model,suffix)
#         for region in regions:
#             mod_path = './models/EINN/exp{}/{}/'.format(experiment,region)
#             if not os.path.exists(mod_path):
#                 os.makedirs(mod_path)
#             load_model(mod_path+file_name,self,region)
#         print('load complete')

#     def forward_feature(self,region,X,X_mask,time_seq):
#         ''' 
#             Feature module forward pass
#         '''
#         X_embeds = self.encoder.mods['encoder_'+region[0]].forward_mask(X.transpose(1, 0), X_mask.transpose(1, 0))
#         time_seq = time_seq[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:,:]
#         Hi_data = (time_seq - self.t_min)/(self.t_max - self.t_min)
#         emb_prime = self.decoder.mods['decoder_'+region[0]](Hi_data,X_embeds)
#         states_prime = self.out_layer.mods['output_'+region[0]](emb_prime) 
#         return states_prime, emb_prime

#     def forward_time(self,region,time_seq):
#         ''' 
#             Time module forward pass
#             Works for a single region at the time 
#         '''
#         inputi_data = time_seq.permute(1,0,2)
#         Hi_data = (inputi_data - self.t_min)/(self.t_max - self.t_min)
#         emb_Ei_data = self.time_nn_mod.time_mods['time_nn_'+region[0]](Hi_data)
#         statesi_data = self.out_layer.mods['output_'+region[0]](emb_Ei_data)

#         return statesi_data.permute(1,0,2), emb_Ei_data.permute(1,0,2)
        
#     def forward_ode(self,region,time_seq):
#         """
#             Compute ds/dt for time module
#         """
#         # pass w/ require grad
#         t_eqns = torch.tensor(time_seq,requires_grad=True)
#         states_eqns, _ = self.forward_time(region,t_eqns)

#         # scale to get gradients
#         states_eqns = self.rmse_scalers[region[0]].inverse_transform(states_eqns)
        
#         ones_tensor = torch.ones_like(t_eqns,device=self.device)
#         stateS_dt = torch.autograd.grad(states_eqns[:,:,[0]],t_eqns,grad_outputs=ones_tensor,create_graph=True) 
#         stateE_dt = torch.autograd.grad(states_eqns[:,:,[1]],t_eqns,grad_outputs=ones_tensor,create_graph=True) 
#         stateI_dt = torch.autograd.grad(states_eqns[:,:,[2]],t_eqns,grad_outputs=ones_tensor,create_graph=True) 
#         stateR_dt = torch.autograd.grad(states_eqns[:,:,[3]],t_eqns,grad_outputs=ones_tensor,create_graph=True) 
#         state_dt = torch.cat((stateS_dt[0], stateE_dt[0], stateI_dt[0], stateR_dt[0]), 2)
#         return states_eqns, state_dt
    
#     def forward_gradient_feat(self,region,X,X_mask,time_seq):
#         """ forward pass to calculate ds/dt for feature module """

#         """ calculate de/dt via time module """
#         t_eqns = time_seq[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:]
#         inputi_eqns = torch.tensor(t_eqns.permute(1,0,2),requires_grad=True)
#         Hi_data = (inputi_eqns - self.t_min)/(self.t_max - self.t_min)
#         emb_Ei_eqns = self.time_nn_mod.time_mods['time_nn_'+region[0]](Hi_data)
#         dEdt = []
#         ones_tensor = torch.ones_like(inputi_eqns,device=self.device)
#         for i in range(emb_Ei_eqns.shape[2]): # size of hidden
#             dEdt.append(torch.autograd.grad(emb_Ei_eqns[:,:,[i]],inputi_eqns,grad_outputs=ones_tensor,create_graph=True)[0])
#         dEdt = torch.cat(dEdt, 2)
#         dEdt = dEdt.permute(1,0,2)
#         if self.detach_yaz:
#             dEdt = dEdt.detach()

#         """ calculate ds/de via feature module """
#         _, emb_E_prime = self.forward_feature(region,X,X_mask,time_seq)
#         # forward pass over the same feature module but saves grad on input
#         # should do scaling as per self.predict_feature 
#         emb_E_grad = torch.tensor(emb_E_prime,requires_grad=True)
#         states_prime = self.out_layer.mods['output_'+region[0]](emb_E_grad) 
#         # scale to get gradients
#         states_prime = self.rmse_scalers[region[0]].inverse_transform(states_prime)
#         ones_tensor = torch.ones_like(states_prime[:,:,4],device=self.device)
#         stateS_de = torch.autograd.grad(states_prime[:,:,0],emb_E_grad,grad_outputs=ones_tensor,create_graph=True)[0] 
#         stateE_de = torch.autograd.grad(states_prime[:,:,1],emb_E_grad,grad_outputs=ones_tensor,create_graph=True)[0] 
#         stateI_de = torch.autograd.grad(states_prime[:,:,2],emb_E_grad,grad_outputs=ones_tensor,create_graph=True)[0] 
#         stateR_de = torch.autograd.grad(states_prime[:,:,3],emb_E_grad,grad_outputs=ones_tensor,create_graph=True)[0] 

#         """ multiply previous two results """
#         stateS_dt = (dEdt * stateS_de).sum(2).unsqueeze(2)
#         stateE_dt = (dEdt * stateE_de).sum(2).unsqueeze(2)
#         stateI_dt = (dEdt * stateI_de).sum(2).unsqueeze(2)
#         stateR_dt = (dEdt * stateR_de).sum(2).unsqueeze(2)
#         gradient_feat = torch.cat((stateS_dt, stateE_dt, stateI_dt, stateR_dt), 2)
#         return states_prime, gradient_feat 

#     def data_loss(self,states_data,y,y_mask,y_w):
#         """ data loss for time module """
#         ys_data_mask = y_mask
#         ys_data_weights = y_w

#         total_data_target_tokens = torch.sum(ys_data_mask != 0).cpu() # denominator of loss

#         criterion = nn.MSELoss(reduction='none')
#         data_loss = \
#                 ((criterion(
#                     states_data[:,:,4],
#                     y[:,:,0]
#                     ) * ys_data_mask * ys_data_weights).sum() )/total_data_target_tokens
#         return data_loss

#     def feat_data_loss(self,states_prime,y,y_mask,y_w):
#         """ data loss for feature module """
#         ys_data_mask = y_mask
#         ys_data_weights = y_w
#         total_data_target_tokens = torch.sum(ys_data_mask != 0).cpu() # denominator of loss
#         criterion = nn.MSELoss(reduction='none')
#         feat_data_loss = (
#             criterion(
#                 states_prime[:,:,4],
#                 y[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:,0]
#             ) * ys_data_mask[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:] * \
#                 ys_data_weights[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:]
#             ).sum()
#         feat_data_loss /= total_data_target_tokens
#         return feat_data_loss

#     def aux_loss(self,states_data,y_mask,rmse_seq):
#         """ 
#             uses analytical data as auxliary aid for learning hidden dynamics 
#         """
#         cols_select = np.array([True, True, True, True, False])
#         ys_data_mask = y_mask.unsqueeze(2).expand(-1,-1,4)
#         total_data_target_tokens = torch.sum(ys_data_mask != 0).cpu() # denominator of loss
#         aux_data = rmse_seq
#         aux_loss = (\
#             F.mse_loss(
#                 states_data[:,:,cols_select],
#                 aux_data[:,:,cols_select],
#                 reduction='none'
#                 ) * ys_data_mask
#             ).sum() / total_data_target_tokens
#         return aux_loss


#     def ode_loss(self,region,states_eqns,state_dt,time_seq,y,y_mask,y_w):
#         """
#             ODE loss for time module
#         """
#         t_eqns = time_seq
#         ys_eqns_mask = y_mask.unsqueeze(2).expand(-1,-1,5) # 4: self.no_ode_states-1
#         ys_eqns_weights = y_w.unsqueeze(2).expand(-1,-1,5)
#         total_eqns_target_tokens = torch.sum(ys_eqns_mask != 0).cpu() # denominator of loss
#         batch_size = t_eqns.shape[0]

#         # scale y
#         ys = self.ys_scalers[region[0]].inverse_transform(y)
        
#         dstate_time_nn = []
#         ode_loss = torch.tensor(0.,dtype=dtype,device=self.device)
#         # we go over each row, which represents a sequence of predictions that follow an ode
#         for rowi in range(batch_size):
#             # I need non-scaled time sequence
#             dstate = []
#             for j, t in enumerate(t_eqns[rowi,:,:]):
#                 ode_id = region[0] + ' ode_'+str(int(t.item()))
#                 dstate.append(self.ode.ode_mods[ode_id].ODE(states_eqns[rowi,[j],:]))
#             dstate = torch.cat(dstate,axis=0)
#             ode_loss_i = (
#                     F.mse_loss(torch.zeros_like(dstate[:,:4]),
#                     (dstate[:,:4]-state_dt[rowi,:,:])/self.S_scale[region[0]][:4],reduction='none') * \
#                         ys_eqns_mask[rowi,:,:4] * ys_eqns_weights[rowi,:,:4] 
#                     ).sum(0)  
#             ode_loss += ode_loss_i.sum()   
#             dstate_time_nn.append(dstate.unsqueeze(0))
#         dstate_time_nn = torch.cat(dstate_time_nn,axis=0)
        
#         # dstate is coming from ode eqns, we want to make dIM to be equal to inc deaths
#         # there is no dIM in autograd outgrad, so we don't need to do it
#         criterion = nn.MSELoss(reduction='none')
#         ode_loss += (criterion(
#             dstate_time_nn[:,:,4] /self.S_scale[region[0]][4],  # S_scale[4] comes from ground truth, do not change to S_grad_scale
#             ys[:,:,0] /self.S_scale[region[0]][4]
#             ) * ys_eqns_mask[:,:,4] * ys_eqns_weights[:,:,4] 
#             ).sum()
        
#         # finally divide by all tokens
#         ode_loss /= (total_eqns_target_tokens)  
#         return ode_loss

#     def future_ode_loss(self,region,states_eqns,state_dt,time_seq,y_mask):
#         """
#             ODE loss in extrapolation domain
#         """
#         t_eqns = time_seq
#         ys_eqns_mask = y_mask.unsqueeze(2).expand(-1,-1,5) # 4: self.no_ode_states-1
#         ys_eqns_mask = torch.ones_like(ys_eqns_mask) - ys_eqns_mask
#         total_eqns_target_tokens = torch.sum(ys_eqns_mask != 0).cpu() # denominator of loss
#         batch_size = t_eqns.shape[0]

#         # if there is no future
#         if total_eqns_target_tokens == 0:
#             return torch.tensor(0.0,dtype=dtype,device=self.device)

#         dstate_time_nn = []
#         ode_loss = torch.tensor(0.,dtype=dtype,device=self.device)
#         # we go over each row, which represents a sequence of predictions that follow an ode
#         ode_id = region[0] + ' ode_'+str(int(self.t_max.item()))
#         for rowi in range(batch_size):
#             dstate = self.ode.ode_mods[ode_id].ODE_detach(states_eqns[rowi,:])
#             ode_loss_i = (
#                     F.mse_loss(torch.zeros_like(dstate[:,:4]),
#                     (dstate[:,:4]-state_dt[rowi,:,:])/self.S_scale[region[0]][:4],reduction='none') * \
#                         ys_eqns_mask[rowi,:,:4] 
#                     ).sum(0)  # element-wise scaling
#             ode_loss += ode_loss_i.sum()   
#             dstate_time_nn.append(dstate.unsqueeze(0))
#         dstate_time_nn = torch.cat(dstate_time_nn,axis=0)

#         # dstate is coming from ode eqns, we want to make dIM to be equal to inc deaths
#         # there is no dIM in autograd outgrad, so we don't need to do it
#         criterion = nn.MSELoss(reduction='none')
#         ode_loss += (
#             criterion(
#                 dstate_time_nn[:,:,4] /self.S_scale[region[0]][4], 
#                 states_eqns[:,:,4] /self.S_scale[region[0]][4]
#             )* ys_eqns_mask[:,:,4] 
#             ).sum()
#         # finally divide by all tokens
#         ode_loss /= (total_eqns_target_tokens)  
#         return ode_loss

#     def kd_loss(self,states_data,states_prime,emb_E,emb_E_prime,y_mask,y_w):
#         """
#             KD loss for outputs and embeddings
#         """
#         if self.detach_yaz:
#             states_data = states_data[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:,:].detach()
#             emb_E = emb_E[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:,:].detach()
#         else:
#             states_data = states_data[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:,:]  # .detach()
#             emb_E = emb_E[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:,:]  # .detach()
#         ys_data_mask = y_mask[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:].unsqueeze(2).expand(-1,-1,5)
#         ys_data_weights = y_w[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:].unsqueeze(2).expand(-1,-1,5)
#         total_data_target_tokens = torch.sum(ys_data_mask != 0).cpu() # denominator of loss
#         criterion = nn.MSELoss(reduction='none')
#         kd_loss_target = (
#             (criterion(
#                 states_prime,  # 0 because feat module only predict that
#                 states_data
#                 ) * ys_data_mask  * ys_data_weights
#             ).sum(2) 
#             ).sum() / total_data_target_tokens
        
#         kd_loss_emb = (
#                 (criterion(
#                     emb_E_prime,
#                     emb_E
#                     ).mean(2) * ys_data_mask[:,:,4] * ys_data_weights[:,:,4] # use one column of mask as we use mean
#                 ) 
#                 ).sum() / (total_data_target_tokens/5)
#         return kd_loss_target, kd_loss_emb

    
#     def feat_ode_loss_mse(self,region,states_feat,feat_dt,time_seq,y,y_mask,y_w):
#         """
#             ODE loss for feature module
#         """
#         t_eqns = time_seq[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:].detach()
#         ys_eqns_mask = y_mask[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:].unsqueeze(2).expand(-1,-1,5)
#         ys_eqns_weights = y_w[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:].unsqueeze(2).expand(-1,-1,5)
#         total_eqns_target_tokens = torch.sum(ys_eqns_mask != 0).cpu() # denominator of loss
#         batch_size = t_eqns.shape[0]

#         # scale y
#         y = self.ys_scalers[region[0]].inverse_transform(y)
        
#         dstate_time_nn = []
#         ode_loss = torch.tensor(0.,dtype=dtype,device=self.device)

#         # we go over each row, which represents a sequence of predictions that follow an ode
#         for rowi in range(batch_size):
#             # I need non-scaled time sequence
#             dstate = []
#             for j, t in enumerate(t_eqns[rowi,:,:]):
#                 ode_id = region[0] + ' ode_'+str(int(t.item()))
#                 dstate.append(self.ode.ode_mods[ode_id].ODE_detach(states_feat[rowi,[j],:]))
#             if self.detach_yaz:
#                 dstate = torch.cat(dstate,axis=0).detach()
#             else:
#                 dstate = torch.cat(dstate,axis=0)
#             ode_loss_i = (
#                     # F.mse_loss(torch.zeros_like(dstate),(dstate-state_dt[rowi,:,:4])/self.S_grad_scale[:4],reduction='none') * ys_eqns_mask[rowi,:].expand(self.no_ode_states-1, -1).transpose(1,0)
#                     F.mse_loss(torch.zeros_like(dstate[:,:4]),
#                     (dstate[:,:4]-feat_dt[rowi,:,:])/self.S_scale[region[0]][:4],reduction='none') * 
#                     ys_eqns_mask[rowi,:,:4] * ys_eqns_weights[rowi,:,:4] 
#                     ).sum(0) 
#             ode_loss += ode_loss_i.sum()   
#             dstate_time_nn.append(dstate.unsqueeze(0))
#         dstate_time_nn = torch.cat(dstate_time_nn,axis=0)

#         # dstate is coming from ode eqns, we want to make dIM to be equal to inc deaths
#         # there is no dIM in autograd outgrad, so we don't need to do it
#         criterion = nn.MSELoss(reduction='none')
#         ode_loss += (
#             criterion(
#                 dstate_time_nn[:,:,4] /self.S_scale[region[0]][4], 
#                 y[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:,0] /self.S_scale[region[0]][4]
#             ) * ys_eqns_mask[:,:,4] * ys_eqns_weights[:,:,4] 
#             ).sum()
#         # finally divide by all tokens
#         ode_loss /= (total_eqns_target_tokens)  
#         return ode_loss

#     def future_feat_ode_loss(self,region,states_eqns,state_dt,time_seq,y_mask):
#         """
#             ODE loss in extrapolation domain
#         """

#         t_eqns = time_seq[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:].detach()
#         ys_eqns_mask = y_mask[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:].unsqueeze(2).expand(-1,-1,5)
#         total_eqns_target_tokens = torch.sum(ys_eqns_mask != 0).cpu() # denominator of loss
#         batch_size = t_eqns.shape[0]

#         # if there is no future
#         if total_eqns_target_tokens == 0:
#             return torch.tensor(0.0,dtype=dtype,device=self.device)

#         dstate_time_nn = []
#         ode_loss = torch.tensor(0.,dtype=dtype,device=self.device)
#         # we go over each row, which represents a sequence of predictions that follow an ode
#         ode_id = region[0] + ' ode_'+str(int(self.t_max.item()))
#         for rowi in range(batch_size):
#             dstate = self.ode.ode_mods[ode_id].ODE_detach(states_eqns[rowi,:])
#             ode_loss_i = (
#                     F.mse_loss(torch.zeros_like(dstate[:,:4]),
#                     (dstate[:,:4]-state_dt[rowi,:,:])/self.S_scale[region[0]][:4],reduction='none') * \
#                         ys_eqns_mask[rowi,:,:4] 
#                     ).sum(0)  # element-wise scaling
#             ode_loss += ode_loss_i.sum()   
#             dstate_time_nn.append(dstate.unsqueeze(0))
#         dstate_time_nn = torch.cat(dstate_time_nn,axis=0)
#         # dstate is coming from ode eqns, we want to make dIM to be equal to inc deaths
#         # there is no dIM in autograd outgrad, so we don't need to do it
#         criterion = nn.MSELoss(reduction='none')
#         ode_loss += (
#             criterion(
#                 dstate_time_nn[:,:,4] /self.S_scale[region[0]][4], 
#                 states_eqns[:,:,4] /self.S_scale[region[0]][4]
#             )* ys_eqns_mask[:,:,4] 
#             ).sum()
#         # finally divide by all tokens
#         ode_loss /= (total_eqns_target_tokens)  
#         return ode_loss

#     def ode_param_loss(self,region,time_seq,y_mask):
#         """ 
#             ODE parameter loss aims to make 
#                 these parameters change smoothly 
#         """
#         t_eqns = time_seq.detach()
#         ys_eqns_mask = y_mask
#         total_eqns_target_tokens = torch.sum(ys_eqns_mask != 0).cpu() # denominator of loss
#         batch_size = t_eqns.shape[0]
#         """
#         """
#         ode_param_loss = torch.tensor(0.,dtype=dtype,device=self.device)
#         criterion = nn.MSELoss(reduction='mean')
#         # we go over each row, which represents a sequence of predictions that follow an ode
#         for rowi in range(batch_size):
#             prev_ode_param = None
#             for j, t in enumerate(t_eqns[rowi,:,:]):
#                 # do not consider the ones that do not belong to the sequence of interest
#                 if ys_eqns_mask[rowi,j]==0:
#                     break
#                 ode_id = region[0] + ' ode_'+str(int(t.item())) 
#                 ode_param = self.ode.ode_mods[ode_id].get_param_vector()
#                 if prev_ode_param is not None:
#                     ode_param_loss += criterion(prev_ode_param,ode_param)
#                 else:
#                     prev_ode_param = ode_param
#         ode_param_loss /= total_eqns_target_tokens
#         return ode_param_loss

#     def monotonicity_loss(self,region,state_dt,y_mask):
#         """
#             calculates monotonicity loss 

#             @param state_dt: autograd derivative
#                 is coming from forward ode
#         """
#         ys_eqns_mask = y_mask
#         # consider also future:
#         ys_eqns_mask = torch.ones_like(ys_eqns_mask)
#         total_eqns_target_tokens = torch.sum(ys_eqns_mask != 0).cpu()
#         # Susceptible monotonically decreases
#         # we want ds/dt <= 0
#         # then, min ds/dt * relu(ds/dt)
#         # penalizes when ds/dt is positive
#         monotonicity_loss = state_dt[:,:,0] * F.relu(state_dt[:,:,0]) /self.S_scale[region[0]][0] * ys_eqns_mask 
#         # Recovered monotonically increases
#         # we want dR/dt >= 0
#         # then, min -dR/dt * relu(-dR/dt)
#         # penalizes when dR/dt is negative
#         monotonicity_loss += -1*state_dt[:,:,3] * F.relu(-1*state_dt[:,:,3]) /self.S_scale[region[0]][3] * ys_eqns_mask
        
#         monotonicity_loss = monotonicity_loss.sum() / (2*total_eqns_target_tokens) 
#         return monotonicity_loss

#     def set_ode_static(self):
#         for param in self.ode.parameters():
#             param.requires_grad = False

#     def set_ode_trainable(self):
#         for param in self.ode.parameters():
#             param.requires_grad = True

#     def set_all_trainable(self):
#         self.feat_nn_mod.train()
#         self.ode.train()
#         for param in self.feat_nn_mod.parameters():
#             param.requires_grad = True
#         for param in self.time_nn_mod.parameters():
#             param.requires_grad = True
#         for param in self.ode.parameters():
#             param.requires_grad = True

#     def set_yaz_static(self):
#         self.set_time_nn_static()
#         self.set_ode_static()

#     def set_time_nn_static(self):
#         self.time_nn_mod.eval()
#         for param in self.time_nn_mod.parameters():
#             param.requires_grad = False

#     def set_time_nn_trainable(self):
#         self.time_nn_mod.eval()
#         for param in self.time_nn_mod.parameters():
#             param.requires_grad = True

#     def set_out_layer_static(self):
#         self.out_layer.eval()
#         for param in self.out_layer.parameters():
#             param.requires_grad = False

#     def set_out_layer_trainable(self):
#         self.out_layer.eval()
#         for param in self.out_layer.parameters():
#             param.requires_grad = True

#     def set_feat_static(self):
#         self.encoder.eval()
#         self.decoder.eval()
#         for param in self.encoder.parameters():
#             param.requires_grad = False
#         for param in self.decoder.parameters():
#             param.requires_grad = False
#         self.rnn_static = True

#     def set_yaz_trainable(self):
#         self.time_nn_mod.train()
#         self.ode.train()
#         for param in self.time_nn_mod.parameters():
#             param.requires_grad = True
#         for param in self.ode.parameters():
#             param.requires_grad = True
#         self.yaz_static = False

#     def set_feat_trainable(self):
#         self.encoder.train()
#         self.decoder.train()
#         for param in self.encoder.parameters():
#             param.requires_grad = True
#         for param in self.decoder.parameters():
#             param.requires_grad = True
#         self.rnn_static = False

#     def train_only_rnn(self):
#         self.set_feat_static()
#         self.set_feat_trainable()

#     def get_hyperparameters(self):
#         """ using this for odenn and baselines"""
#         model_params_path = './setup/'
#         if not os.path.exists(model_params_path):
#             os.makedirs(model_params_path)
#         best_model_params_json_file = model_params_path+'EINN-params.json'
#         # read best hyperparams found for region
#         if os.path.exists(best_model_params_json_file):
#             print('test mode, using existing params json')
#             self.read_hyperparams_from_json(best_model_params_json_file) 
#         else:
#             raise Exception(f'no setup file {best_model_params_json_file}')

#     def read_hyperparams_from_json(
#         self,
#         model_params_json_file
#         ):
#         """ Reads hyperparameters for each, this are found in validation set"""

#         # read from json
#         with open(model_params_json_file) as f:
#             self.model_metadata = json.load(f)
#         #Initialize model with hyperparameters.
#         self.num_epochs = self.model_metadata['NUM_EPOCHS']
#         self.keep_training = self.model_metadata['KEEP_TRAINING']
#         self.lr = self.model_metadata['LEARNING_RATE']
#         self.loss_weights = self.model_metadata['LOSS_WEIGHTS']
#         self.init_ode_wcalibration = self.model_metadata['INIT_ODE']
#         self.batch_size = self.model_metadata['BATCH_SIZE']

#     def set_feat_flag(self):
#         # save epoch where we change to feat module
#         self.feat_flags.append(self.epoch)

#     def plot_loss(self,suffix2=''):
#         path = './figures/exp{}/'.format(self.exp)
#         if not os.path.exists(path):
#             os.makedirs(path)
#         loss_names = ['total_loss','ode_loss','f_ode_loss','data_loss','aux_loss','ode_param_loss']
#         losses = [self.losses,self.ode_losses,self.future_ode_losses,self.data_losses,self.aux_losses,self.ode_param_losses]
#         # if self.feat_mode:
#         loss_names.extend(['kd_target','kd_emb','f_data_loss','gradient_losses'])
#         losses.extend([self.kd_target_losses, self.kd_emb_losses, self.feat_data_losses, self.gradient_losses])
#         time = [i for i in range(len(losses[0]))]  
#         for loss_name, loss in zip(loss_names,losses):
#             plt.yscale('log')
#             plt.xlabel('epochs')
#             plt.plot(time,loss,label=loss_name)
#         if len(self.train_start_flag)>0:
#             for flag in self.train_start_flag:
#                 plt.axvline(x = flag,color='y',linewidth=1,linestyle="dashed")
#         plt.legend()
#         i = 0
#         if self.region=='all':
#             figname = f'losses-{self.ode_model}-{self.pred_week}-{self.exp}-{suffix2}-{i}.png'
#         else:
#             regions_str = '-'.join(self.region)
#             figname = f'{regions_str}_losses-{self.ode_model}-{self.pred_week}-{self.exp}-{suffix2}-{i}.png'
#         while os.path.exists(path+figname):
#             if self.region=='all':
#                 figname = f'losses-{self.ode_model}-{self.pred_week}-{self.exp}-{suffix2}-{i}.png'
#             else:
#                 regions_str = '-'.join(self.region)
#                 figname = f'{regions_str}_losses-{self.ode_model}-{self.pred_week}-{self.exp}-{suffix2}-{i}.png'
#             i += 1
#         plt.savefig(path+figname)
#         plt.close()

#     def save_predictions(
#         self,
#         region: str,
#         death_predictions: np.array,
#         submodule=None,
#         ):
#         """
#             Given an array w/ predictions, save as csv
#         """
#         data = np.array(
#             [
#                 np.arange(len(death_predictions))+1,
#                 death_predictions
#             ]
#         )
#         df = pd.DataFrame(data.transpose(),columns=['k_ahead','deaths'])
#         df['k_ahead'] = df['k_ahead'].astype('int8')
#         path = './results/COVID/{}/'.format(region)
#         if not os.path.exists(path):
#             os.makedirs(path)
#         model_name = 'EINN'+submodule
#         file_name = 'preds_{}_{}_exp{}.csv'.format(model_name,self.pred_week,self.exp)
#         df.to_csv(path+file_name,index=False)

#     def predict_save(self,suffix=''):
        
#         self.eval()
#         with torch.no_grad():  #saves memory
#             # only one batch
#             region, X, X_mask, _, _, _, _, time_seq = next(iter(self.test_loader)) 
#             self.eval()
#             X = X.to(self.device, non_blocking=True)
#             X_mask = X_mask.to(self.device, non_blocking=True)
#             time_seq = time_seq.to(self.device, non_blocking=True)

#             for i in range(len(region)):
#                 print('predict in ',region[i])
#                 states_prime, _ = self.forward_feature([region[i]],X[[i]],X_mask[[i]],time_seq[[i]])
#                 death_pred_feat = states_prime[:,:,4].reshape(1,-1)
#                 death_pred_feat = self.ys_scalers[region[i]].inverse_transform(death_pred_feat)
#                 death_pred_feat = death_pred_feat.reshape(-1).detach().cpu().data.numpy()
#                 self.save_predictions(region[i],death_pred_feat,''+suffix)

    
#     # https://github.com/AdityaLab/CAMul/blob/master/train_covid.py#L284
#     def evaluate(self):
#         # evaluates in training
#         total_mse_error = []
#         with torch.no_grad():
#             # go over region because architecture depends on it
#             for r in np.random.permutation(regions): 
#                 region, X, X_mask, y, y_mask, _, _, time_seq = next(iter(self.data_loaders[r])) 
#                 self.eval()
#                 X = X.to(self.device, non_blocking=True)
#                 X_mask = X_mask.to(self.device, non_blocking=True)
#                 time_seq = time_seq.to(self.device, non_blocking=True)
#                 y = y.to(self.device, non_blocking=True)
#                 y_mask = y_mask.to(self.device, non_blocking=True)

#                 # forward feature module 
#                 states_prime, _ = self.forward_feature(region,X,X_mask,time_seq)

#                 ys_data_mask = y_mask
#                 total_data_target_tokens = torch.sum(ys_data_mask != 0).cpu() # denominator of loss
#                 criterion = nn.MSELoss(reduction='none')
#                 mse_error = (
#                     criterion(
#                         states_prime[:,:,4],
#                         y[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:,0]
#                     ) * ys_data_mask[:,-WEEKS_AHEAD*DAY_WEEK_MULTIPLIER:]
#                     ).sum()
                
#                 mse_error /= total_data_target_tokens
#                 total_mse_error.append(mse_error.cpu().item())

#         rmse_error = np.sqrt(np.array(total_mse_error).mean())
#         return rmse_error


#     def _train(self,epochs,time_reps,time_ode_reps,feat_time_reps,out_reps):

#         self.train_start_flag.append(self.epoch)  # save epoch where we start this training, used in loss plot

#         ############## optimizers #############
#         # time nn only
#         time_params = itertools.chain(self.time_nn_mod.parameters(),self.out_layer.parameters()) 
#         optimizer_time = torch.optim.Adam(time_params, lr=self.lr, amsgrad=True)
#         # time nn + ode only
#         time_ode_params = itertools.chain(self.time_nn_mod.parameters(),self.ode.parameters(),\
#             self.out_layer.parameters())  
#         optimizer_time_ode = torch.optim.Adam(time_ode_params, lr=self.lr, amsgrad=True)
#         # time nn + ode + feature
#         time_feat_params = itertools.chain(self.encoder.parameters(),self.decoder.parameters(),\
#             self.out_layer.parameters(),self.time_nn_mod.parameters())  
#         optimizer_time_feat = torch.optim.Adam(time_feat_params, lr=self.lr, amsgrad=True)
#         # output layers + ode params
#         out_ode_params = itertools.chain(self.ode.parameters(),self.out_layer.parameters()) 
#         optimizer_out_ode = torch.optim.Adam(out_ode_params, lr=self.lr)
#         # will search for 1% improvement at least
#         es = EarlyStopping(patience=20, min_delta=10, percentage=True)  # patience was 25

#         self.epoch = 0
#         self.train_start_flag.append(self.epoch)  # save epoch where we start this training, used in loss plot
         
#         self.best_loss = 10e3


#         rmse_val = self.evaluate()
#         print('initial validation rmse', rmse_val)
#         for epoch in range(epochs):

#             """ time solo """
#             for _ in range(time_reps):
#                 self.train_time = True
#                 self.train_feat = False
#                 self.use_ode_loss = False
#                 self.add_grad_loss = False
#                 self.detach_yaz = True  # True: detach both time and ode
#                 self.set_ode_static()
#                 self.set_feat_static()
#                 self.set_time_nn_trainable()
#                 self.set_out_layer_trainable()
#                 self.minibatch_train(optimizer_time,time_params)

#             """ time + ode """
#             for _ in range(time_ode_reps):
#                 self.train_yaz = True
#                 self.train_feat = False
#                 self.use_ode_loss = True
#                 self.add_grad_loss = False
#                 self.detach_yaz = True  # True: detach both time and ode
#                 self.set_feat_static()
#                 self.set_time_nn_trainable()
#                 self.set_ode_trainable()
#                 self.set_out_layer_trainable()
#                 self.minibatch_train(optimizer_time_ode,time_ode_params)
           
#             """ feat and time jointly """
#             for _ in range(feat_time_reps):
#                 self.train_time = True
#                 self.train_feat = True
#                 self.use_ode_loss = True
#                 self.add_grad_loss = False
#                 self.detach_yaz = False  # True: detach both time and ode
#                 self.set_ode_trainable()
#                 self.set_feat_trainable()
#                 self.set_time_nn_trainable()
#                 self.set_out_layer_trainable()
#                 self.minibatch_train(optimizer_time_feat,time_feat_params)

#             """ only out layer + ode params """
#             for _ in range(out_reps):
#                 self.train_time = True
#                 self.train_feat = True
#                 self.use_ode_loss = True
#                 self.add_grad_loss = True
#                 self.detach_yaz = False  # True: detach both time and ode
#                 self.set_time_nn_static()
#                 self.set_ode_trainable()
#                 self.set_feat_static()
#                 self.set_out_layer_trainable()
#                 self.minibatch_train(optimizer_out_ode,out_ode_params)

#             ''' save model '''
#             rmse_val = self.evaluate()
#             save_model = True

#             print(f'Train RMSE: {rmse_val}, bad: {es.num_bad_epochs}' )
            
#             if rmse_val < self.best_loss:
#                 print(f'>> best updated {self.exp} on {self.pred_week}, epoch {epoch}')
#                 self.best_loss = torch.tensor(rmse_val)
#                 if save_model:
#                     self.save_odernn_local(suffix='-post')
#             ''' early stopping ''' 
#             if self.early:
#                 if self.epoch > self.patience_before_es:
#                     if es.step(self.best_loss):
#                         print('====BREAK, early stopping ===')
#                         break 


#     def minibatch_train(self,optims,params,verbose=True):

#         self.epoch += 1

#         epoch_total_loss = []
#         epoch_total_loss = [] 
#         epoch_ode_loss = [] 
#         epoch_future_ode_loss = [] 
#         epoch_data_loss = [] 
#         epoch_kd_target_loss = [] 
#         epoch_kd_emb_loss = [] 
#         epoch_feat_data_loss = [] 
#         epoch_ode_param_loss = [] 
#         epoch_monotonicity_loss = [] 
#         epoch_feat_ode_loss = [] 
#         epoch_future_feat_ode_loss = [] 
#         epoch_aux_loss = [] 
#         i = 0
#         self.train()
#         start_time = time.time()
#         optims.zero_grad(set_to_none=True)
#         for r in np.random.permutation(regions):  # one region at the time
#             # backprop = False
#             region, X, X_mask, y, y_mask, y_w, rmse_seq, time_seq = next(iter(self.data_loaders[r])) 
            
#             X = X.to(self.device, non_blocking=True)
#             X_mask = X_mask.to(self.device, non_blocking=True)
#             time_seq = time_seq.to(self.device, non_blocking=True)
#             y = y.to(self.device, non_blocking=True)
#             y_mask = y_mask.to(self.device, non_blocking=True)
#             y_w = y_w.to(self.device, non_blocking=True)
#             rmse_seq = rmse_seq.to(self.device, non_blocking=True)

#             # forward feature module 
#             states_data, emb = self.forward_time(region,time_seq)

#             if self.train_time:
                
#                 data_loss = self.data_loss(states_data,y,y_mask,y_w)
#                 aux_loss = self.aux_loss(states_data,y_mask,rmse_seq)

#                  # sample time sequence
#                 idx = np.random.choice(np.arange(time_seq.shape[0]),3)
#                 eqns_time_seq, eqns_y, eqns_y_mask, eqns_y_w = time_seq[idx], y[idx],y_mask[idx],y_w[idx]
#                 states_eqns, state_dt = self.forward_ode(region,eqns_time_seq)
#                 ''' ODE loss '''
#                 if self.use_ode_loss:
#                     # note: states_eqns is scaled to real dimensions
#                     ode_loss = self.ode_loss(region,states_eqns,state_dt,eqns_time_seq,eqns_y,eqns_y_mask,eqns_y_w)
#                     future_ode_loss = self.future_ode_loss(region,states_eqns,state_dt,eqns_time_seq,eqns_y_mask)
#                     ''' Param consistency '''
#                     ode_param_loss = self.ode_param_loss(region,eqns_time_seq,eqns_y_mask)
#                 else:
#                     ode_loss = torch.tensor(0.0,dtype=dtype,device=self.device)
#                     future_ode_loss = torch.tensor(0.0,dtype=dtype,device=self.device)
#                     ode_param_loss = torch.tensor(0.0,dtype=dtype,device=self.device)
                
#                 ''' Monotonicity loss '''
#                 monotonicity_loss = self.monotonicity_loss(region,state_dt,eqns_y_mask)
                
#                 total_loss = self.loss_weights['data']*data_loss + \
#                         self.loss_weights['aux']*aux_loss +\
#                         self.loss_weights['ode']*ode_loss +\
#                         self.loss_weights['future_ode']*future_ode_loss +\
#                         self.loss_weights['ode_param']*ode_param_loss +\
#                             self.loss_weights['monotonicity']*monotonicity_loss
#             else:
#                 data_loss = torch.tensor(0.0,dtype=dtype,device=self.device)
#                 aux_loss = torch.tensor(0.0,dtype=dtype,device=self.device)
#                 monotonicity_loss = torch.tensor(0.0,dtype=dtype,device=self.device)
#                 ode_loss = torch.tensor(0.0,dtype=dtype,device=self.device)
#                 future_ode_loss = torch.tensor(0.0,dtype=dtype,device=self.device)
#                 ode_param_loss = torch.tensor(0.0,dtype=dtype,device=self.device)
#                 total_loss = torch.tensor(0.0,dtype=dtype,device=self.device)

#             ''' KD loss '''
#             if self.train_feat:
#                 # forward feature module 
#                 states_prime, emb_prime = self.forward_feature(region,X,X_mask,time_seq)
#                 feat_data_loss = self.feat_data_loss(states_prime,y,y_mask,y_w)
#                 kd_loss_target, kd_loss_emb = self.kd_loss(states_data,states_prime,emb,emb_prime,y_mask,y_w)
#                 kd_loss_emb = torch.tensor(0.0,dtype=dtype,device=self.device)

#                 total_loss += self.loss_weights['feat_data']*feat_data_loss +\
#                             self.loss_weights['kd_target']*kd_loss_target +\
#                                 self.loss_weights['kd_emb']* kd_loss_emb 
#             else:
#                 feat_data_loss = torch.tensor(0.0,dtype=dtype,device=self.device)
#                 kd_loss_target = torch.tensor(0.0,dtype=dtype,device=self.device)
#                 kd_loss_emb = torch.tensor(0.0,dtype=dtype,device=self.device)

#             ''' Feat ODE loss'''
#             ''' aka feat grad matching '''
#             if self.add_grad_loss:  # use it when embedding loss is small
#                 idx = np.random.choice(np.arange(time_seq.shape[0]),16)
#                 grad_time_seq, grad_X, grad_X_mask = time_seq[idx],X[idx],X_mask[idx]
#                 grad_y, grad_y_mask, grad_y_w =y[idx],y_mask[idx],y_w[idx]
#                 states_feat, feat_dt = self.forward_gradient_feat(region,grad_X,grad_X_mask,grad_time_seq)
#                 feat_ode_loss = self.feat_ode_loss_mse(region,states_feat,feat_dt,grad_time_seq,grad_y,grad_y_mask,grad_y_w)
#                 future_feat_ode_loss = \
#                     self.future_feat_ode_loss(region,states_feat,feat_dt,grad_time_seq,grad_y_mask)
#                 total_loss += self.loss_weights['gradient_loss']*feat_ode_loss +\
#                             self.loss_weights['gradient_loss']*future_feat_ode_loss
#             else:
#                 feat_ode_loss = torch.tensor(0.0,dtype=dtype,device=self.device)
#                 future_feat_ode_loss = torch.tensor(0.0,dtype=dtype,device=self.device)
#             total_loss.backward(retain_graph=False)
#             torch.nn.utils.clip_grad_norm_(params, 10)
#             optims.step()
#             optims.zero_grad(set_to_none=True)

#             epoch_total_loss.append(total_loss.detach().cpu().item())
#             epoch_ode_loss.append(ode_loss.detach().cpu().item())
#             epoch_future_ode_loss.append(future_ode_loss.detach().cpu().item())
#             epoch_data_loss.append(data_loss.detach().cpu().item())
#             epoch_kd_target_loss.append(kd_loss_target.detach().cpu().item())
#             epoch_kd_emb_loss.append(kd_loss_emb.detach().cpu().item())
#             epoch_feat_data_loss.append(feat_data_loss.detach().cpu().item())
#             epoch_ode_param_loss.append(ode_param_loss.detach().cpu().item())
#             epoch_monotonicity_loss.append(monotonicity_loss.detach().cpu().item())
#             epoch_feat_ode_loss.append(feat_ode_loss.detach().cpu().item())
#             epoch_future_feat_ode_loss.append(future_feat_ode_loss.detach().cpu().item())
#             epoch_aux_loss.append(aux_loss.detach().cpu().item())


#         epoch_total_loss = np.array(epoch_total_loss).mean()
#         epoch_ode_loss = np.array(epoch_ode_loss).mean()
#         epoch_future_ode_loss = np.array(epoch_future_ode_loss).mean()
#         epoch_data_loss = np.array(epoch_data_loss).mean()
#         epoch_kd_target_loss = np.array(epoch_kd_target_loss).mean()
#         epoch_kd_emb_loss = np.array(epoch_kd_emb_loss).mean()
#         epoch_feat_data_loss = np.array(epoch_feat_data_loss).mean()
#         epoch_ode_param_loss = np.array(epoch_ode_param_loss).mean()
#         epoch_monotonicity_loss = np.array(epoch_monotonicity_loss).mean()
#         epoch_feat_ode_loss = np.array(epoch_feat_ode_loss).mean()
#         epoch_future_feat_ode_loss = np.array(epoch_future_feat_ode_loss).mean()
#         epoch_aux_loss = np.array(epoch_aux_loss).mean()

#         elapsed = time.time() - start_time

#         if verbose:
#             print('Epoch: %d, Total: %.2e, Data-T: %.2e, ODE: %.2e, ODE-Fut: %.2e, Data-F: %.2e, ODE-F: %.2e, ODE-F-Fut: %.2e, KD-T: %.2e, KD-E: %.2e, Aux: %.2e, Mono: %.2e, Time: %.3f'
#                     %(self.epoch, epoch_total_loss.item(), epoch_data_loss.item(), epoch_ode_loss.item(), epoch_future_ode_loss.item(), \
#                         epoch_feat_data_loss.item(), epoch_feat_ode_loss.item(), epoch_future_feat_ode_loss.item(),  epoch_kd_target_loss.item(), epoch_kd_emb_loss.item(), epoch_aux_loss.item(), \
#                             epoch_monotonicity_loss.item(), elapsed))

#         ''' save losses '''
#         self.losses.append(epoch_total_loss)
#         self.ode_losses.append(epoch_ode_loss)
#         self.data_losses.append(epoch_data_loss)
#         self.kd_target_losses.append(epoch_kd_target_loss)
#         self.kd_emb_losses.append(epoch_kd_emb_loss)
#         self.feat_data_losses.append(epoch_feat_data_loss)
#         self.ode_param_losses.append(epoch_ode_param_loss)
#         self.monotonicity_losses.append(epoch_monotonicity_loss)
#         self.future_ode_losses.append(epoch_future_ode_loss)
#         self.gradient_losses.append(epoch_feat_ode_loss)
#         self.aux_losses.append(epoch_aux_loss)


#     def is_week(self,week):
#         return self.pred_week == week

#     def is_feat_mod_available(self):
#         return os.path.exists(self.mod_path+self.rnnode_file_name + "_encoder.pth")

#     def train_predict(self):

#         # check if feat module is already there
#         try:
#             if self.keep_training:
#                 pass
#             else:
#                 self.predict_save()
#                 return 
#         except:
#             raise Exception('no model loaded')

#         """  train time nn + ode """
#         self.train_time_ode()

#         """  train time + ode + feat """
#         self.train_time_ode_feat()
        
#         """  add ODE-F and train output layer """
#         self.train_out_layers()

#         """ save predictions """
#         self.predict_save()
#         # plot training losses
#         self.plot_loss()
#         return 

#     def train_time_ode(self):
#         ''' alternate training avoids messing up the ODE initialization '''

#         print('\n====== train time nn + ode =======')
#         EPOCHS = self.num_epochs
#         self._train(epochs=EPOCHS,time_reps=1,time_ode_reps=1,feat_time_reps=0,out_reps=0)  
#         return 

#     def train_time_ode_feat(self):
#         ''' resets ode, trains time + feature'''
        
#         print('\n====== train time + ode + feat =======')
#         EPOCHS = self.num_epochs
#         self._train(epochs=EPOCHS,time_reps=0,time_ode_reps=0,feat_time_reps=1,out_reps=0)  
#         return 

#     def train_out_layers(self):

#         print('\n====== train out layer + ode params =======')
#         EPOCHS = self.num_epochs  
#         self._train(epochs=EPOCHS,time_reps=0,time_ode_reps=0,feat_time_reps=0,out_reps=1)  

#         return 