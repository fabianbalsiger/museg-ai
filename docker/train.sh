#!/bin/bash

nnUNetv2_plan_and_preprocess -d $1 --verify_dataset_integrity

nnUNetv2_train $1 $2 0
nnUNetv2_train $1 $2 1
nnUNetv2_train $1 $2 2
nnUNetv2_train $1 $2 3
nnUNetv2_train $1 $2 4