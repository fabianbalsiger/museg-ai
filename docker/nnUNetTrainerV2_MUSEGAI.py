"""Custom nnU-Net trainer allowing to ignore unsegmented image slices in a volume when computing the loss."""
from __future__ import annotations

import SimpleITK as sitk
import numpy as np
import json
import torch
from pathlib import Path
import torchvision.transforms.functional as F

import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

import batchgenerators.utilities.file_and_folder_operations as ffops
import nnunet.training.network_training.nnUNetTrainerV2 as trainerV2
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss



class DC_and_CE_loss_improved(DC_and_CE_loss):
    """Wrapper of the DC_and_CE_loss that does return 0 instead of NaN when the target only consists of the ignored label.

    The problem of the NaN arises due to a division by zero at the line 'ce_loss = ce_loss.sum() / mask.sum()'.
    In such a case, the mask.sum() is simply zero as the target consists only of ignored labels.
    """

    def forward(self, net_output, target):
        """Calculate the Dice and cross-entropy loss."""
        if self.ignore_label is not None and target.unique().size(dim=0) == 1:
            if self.ignore_label == target.unique():
                return 0
        return super().forward(net_output, target)


class nnUNetTrainerV2_MUSEGAI(trainerV2.nnUNetTrainerV2):
    """Custom nnUNetTrainer that ignores the background allowing to train a full 3-D volume segmentation with sparse annotations.

    See Also: Çiçek et al., 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation, MICCAI 2016, https://arxiv.org/abs/1606.06650
    """

    def __init__(
        self,
        plans_file,
        fold,
        output_folder=None,
        dataset_directory=None,
        batch_dice=True,
        stage=None,
        unpack_data=True,
        deterministic=True,
        fp16=False,
    ):
        """Initialize the nnU-Net trainer for muscle segmentation."""
        super().__init__(
            plans_file,
            fold,
            output_folder,
            dataset_directory,
            batch_dice,
            stage,
            unpack_data,
            deterministic,
            fp16,
        )
        if Path(self.plans_file).is_file():
            # during training, we do have access to the plans_file
            plans = ffops.load_pickle(self.plans_file)
        else:
            # during inference, we might not have access to the plans_file
            plans = {"num_classes": None}

        # ignore the background label for training with every n-th image slice in 3-D volumes
        self.loss = DC_and_CE_loss_improved(
            {"batch_dice": self.batch_dice, "smooth": 1e-5, "do_bg": False},
            {},
            ignore_label=plans["num_classes"],
        )

class interactive_nnUNetTrainer(nnUNetTrainer.nnUNetTrainer):
    """custom nnUNet Trainer that train also for interactive segmentation and prediction refinement"""

    def __init__(self, 
                plans: dict, 
                configuration: str, 
                fold: int, 
                dataset_json: dict,
                unpack_dataset: bool = True,
                device: torch.device = torch.device('cuda'),
                #max_iter=5, 
                #nbr_supervised=0.5,
            ):
            """Initialize the nnU-Net trainer for muscle segmentation."""
            super().__init__(plans,
            configuration,
            fold,
            dataset_json,
            unpack_dataset,
            device,
            )

            self.max_iter = 2#max_iter # maximal number of click during an iteration of training
            self.nbr_supervised = 0.5#nbr_supervised  # number of image that are trained with clicks
            self.dataset_json = dataset_json #info on the dataset 
            

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        #Part where we are going to simulate clicks:
        #starting by creating channels to store clicks:
        b,c,d,h,w= data.size()  # batch, channel, depth, height, width
        groundtruth=target[0]

        #get the number of labels

        if 'ignore_label' in self.dataset_json['labels'].keys():
            nbr_labels=len(self.dataset_json['labels'])-1 #-1 because we don't count the ignore label
        else:
            nbr_labels=len(self.dataset_json['labels'])
        
        #click channel are potentialy noised by the preprocessing we need to fix that : 
        if data[:,1:nbr_labels+1].sum()!=0:
            data[:,1:nbr_labels+1]=torch.full((b,c-1,d,h,w),0,device=self.device,dtype=torch.int)

        #function to decide if we simulate the k-th click
        def do_simulate(k,N):
            return np.random.binomial(n=1,p=1-k/N)
        
        if np.random.binomial(n=1,p=self.nbr_supervised): #choosing if the batch is gonna be train with clicks or not
            self.network.eval() #putting the model in inference mode, needed to simulate click
                
            for k in range(self.max_iter):
            #we first want to get map probabilities
                if do_simulate(k,self.max_iter):
                    # using current network to have prediction & probabilities 
                    with torch.no_grad(): 
                        logits = self.network(data)
                        probabilities = torch.softmax(logits[0],dim=1)
                        _,prediction = torch.max(probabilities,dim=1)
                    for nimage in range(b):
                        
                        test = groundtruth[nimage][0] == prediction[nimage] #test matrix to find prediction's mistakes
                        misslabeled_indices = torch.nonzero(~test) #getting indexes of misslabeled pixels
                        
                        for slice in range(d):

                            #checking if the current slice is not fully blank
                            if torch.nonzero(data[nimage,0,slice]-data[nimage,0,slice,0,0]).numel==0:
                                print('slice fully blank')
                                continue

                            #filtering by slice                        
                            mask=(misslabeled_indices[:,0]==slice)
                            misslabeled_per_slice=misslabeled_indices[mask]
                            #getting gt value of all wrongly predicted pixels
                            slice_indices=misslabeled_per_slice[:,0]
                            h_indices=misslabeled_per_slice[:,1]
                            w_indices=misslabeled_per_slice[:,2]
                            true_value=groundtruth[nimage,0][slice_indices,h_indices,w_indices]

                            #getting the worst prediction label
                            label_list,misslabeled_count=true_value.unique(return_counts=True)
                            if len(misslabeled_count)==0: #when the network does 0 mistake because there is nothing to predict
                                print('slice is only noise')
                                continue
                            else:
                                max_value=torch.max(misslabeled_count).item()
                                worst_labels=(misslabeled_count==max_value).nonzero(as_tuple=False)
                                chosen_label=int(label_list[worst_labels][np.random.randint(len(worst_labels))].item())
            
                                #simulation du clique ici
                                mask=(groundtruth[nimage,0,slice]==chosen_label) & (~test[slice])
                                potential_click = torch.nonzero(mask)

                             # choose click with respect to a posteriori probabilites 
                                def chamfer_distance_torch(T1,T2):
                                    """return the tensor containing the distance of from T2 of each point of T1 """
                                    return torch.min(torch.cdist(T1,T2),dim=1)[0]
                                
                                def dilatation(image):
                                    """dilate the "1" part of an 2d image"""
                                    kernel=torch.tensor([[1,1,1],[1,1,1],[1,1,1]],device=self.device).unsqueeze(0).unsqueeze(0).float()
                                    return torch.where(torch.conv2d(image.unsqueeze(0).unsqueeze(0),kernel,padding=1)>=1,1,0).squeeze(0).squeeze(0)
                                
                                def get_probabilities(mask):
                                    """compute chamfer distance and transforms it into probability map (up to a factor)"""
                                    contour=dilatation(mask)-mask
                                    dist=chamfer_distance_torch(torch.nonzero(mask==1).float(),torch.nonzero(contour==1).float())
                                    return torch.exp(dist)-1

                                #filtering groundtruh and prediction to work only where there is the good label                                               
                                gt_label=(groundtruth[nimage,0,slice]==chosen_label).int()
                                pred_label=(prediction[nimage,slice]==chosen_label).int()
                                #computing False negative and False positive masks
                                False_negative=torch.nonzero(torch.where(gt_label-pred_label==1,1,0))
                                False_positive=torch.nonzero(torch.where(gt_label-pred_label==-1,1,0))
                                D_plus=torch.full((h,w),0,device=self.device).float() #False_negative map
                                h_indices=False_negative[:,0]
                                w_indices=False_negative[:,1]
                                D_plus[h_indices,w_indices]=1
                                D_minus=torch.full((h,w),0,device=self.device).float() #False_positive map 
                                h_indices=False_positive[:,0]
                                w_indices=False_positive[:,1]
                                D_minus[h_indices,w_indices]=1
                                
                                #Computing chamfer distance to get probabilities for choosing click's index
                                if D_plus.sum()>=D_minus.sum(): #if we have more False negative we correct one of them
                                   
                                    proba_click=get_probabilities(D_plus)
                                    potential_click=torch.nonzero(proba_click==proba_click.max())
                                    random_pick=potential_click[np.random.randint(len(potential_click))]
                                    click=torch.nonzero(D_plus)[random_pick].squeeze(0)

                                else:
                                  
                                    proba_click=get_probabilities(D_minus)
                                    potential_click=torch.nonzero(proba_click==proba_click.max())
                                    random_pick=potential_click[np.random.randint(len(potential_click))]
                                    click=torch.nonzero(D_minus)[random_pick].squeeze(0)
                                    chosen_label=groundtruth[nimage,0,slice,click[0].item(),click[1].item()].int()
                               
                                #adding click into data                                                              
                                data[nimage,chosen_label+1,slice,click[0],click[1]] = 1
                                                      
                else:
                    break
            #here we smoothed the click data...
            kernel_size=(3,3)
            sigma=(2,2)
            # ... by applying gaussian fitlering and normalization to click's channels
            for channel in range(1,c):
                data[:,channel]=F.gaussian_blur(data[:,channel],kernel_size,sigma)
            print("clicks generated, starting batch training...")
            self.network.train() #putting the model back to training mode 

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
