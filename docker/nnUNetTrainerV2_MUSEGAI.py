"""Custom nnU-Net trainer allowing to ignore unsegmented image slices in a volume when computing the loss."""
from __future__ import annotations

import SimpleITK as sitk
import numpy as np
import json
import torch
from pathlib import Path

import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

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

            self.max_iter = 5#max_iter # maximal number of click during an iteration of training
            self.nbr_supervised = 0.5#nbr_supervised  # number of image that are trained with clicks
            self.dataset_json = dataset_json #info on the dataset -> maybe not useful
            

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        breakpoint()
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        #Part where we are going to simulate clicks:
        #starting by creating channels to store clicks:

        _,d,h,w= data[0].size()

        #get the number of labels

        
        nbr_labels=len(self.dataset_json['labels'])-2 #-2 because we don't count the background label & the ignore label

        # background_T=torch.zeros(1,d,h,w,device='cuda')
        # foreground_T=torch.zeros(1,d*nbr_labels,h,w,device='cuda')

        #function to decide if we simulate the k-th click
        def do_simulate(k,N):
            return np.random.binomial(n=1,p=1-k/N)
        
        self.network.eval() #putting the model in inference mode, needed to simulate click
        for image, groundtruth in zip(data,target[0]):
            inputs=image
            #inputs=torch.cat((image,foreground_T,background_T),axis=1)
            for k in range(self.max_iter):
            #we first want to get map probabilities
                if do_simulate(k,self.max_iter):
                    # using current network to have prediction & probabilities 
                    with torch.no_grad():
                        logits = self.network(data)
                        prediction, probabilities = convert_predicted_logits_to_segmentation_with_correct_shape(logits,plans_manager=self.plans,
                                                                                                                ConfigurationManager=self.Configuration,
                                                                                                                label_manager=self.dataset_json,
                                                                                                                return_probabilities=True)
                    #!!! next line might be not working -> if the prediction is in color and the labels are int. 
                    test = groundtruth == prediction #test matrix to find prediction's mistakes
                    misslabeled_index = np.argwhere(~test) #getting indexes of misslabeled pixels

                    for slice in range(d):
                        misslabeled_count = (nbr_labels+1)*[0] #+1 because we add the background label 
                        for i in [index for index in misslabeled_index if index[0]==slice]:
                            label=groundtruth[tuple(i)]
                            misslabeled_count[label]+=1     

                    #getting the worst predicted label
                    max_value = max(misslabeled_count)
                    worst_labels = [i for i, x in enumerate(misslabeled_count) if x == max_value]
                    if len(worst_labels) != 1:
                        #if there is more than one label, we pick one randomly
                        chosen_label = np.random.choice(worst_labels)
                    else:
                        chosen_label = worst_labels[0]
                    #simulation du clique ici
                    potential_click=[index for index in misslabeled_index if groundtruth[tuple(index)]==chosen_label and index[0]==slice]
                    D={}
                    for coordinate in potential_click:
                        D[coordinate]=probabilities[tuple(coordinate)]
                    max_value = max(D.values)
                    final_list=[]
                    for coordinate in D.items():
                        if coordinate[1] == max_value:
                            final_list.append(coordinate[0])
                    click = np.random.choice(final_list)
                      
                    inputs[tuple(click)] = 1 #a changer si on change les tenseurs foreground et background -> !!! à modifier 
                else:
                    break
                #here we smoothed the click data
                ...
                #writing the simulation for optimisation...
                data[data.index(image)] = inputs
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
