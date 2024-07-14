import math
import numpy as np
import torch.nn as nn
import torch

def split(ids, train, val, test):
    # proportions of train, val, test
    assert (train+val+test == 1)
    IDs = np.unique(ids)
    num_ids = len(IDs)

    # priority given to the test/val sets
    test_split = math.ceil(test * num_ids)
    val_split = math.ceil(val * num_ids)
    train_split = num_ids - val_split - test_split

    train = np.where(np.isin(ids, IDs[:train_split]))[0]
    val = np.where(np.isin(ids, IDs[train_split:train_split+val_split]))[0]
    test = np.where(np.isin(ids, IDs[train_split+val_split:]))[0]
    
    return train, val, test

def compute_distil_loss(t_outputs, s_outputs, targets, s_loss_fn, KLD, temperature=20, lambda_param=0.9):
    # compute soft_targets
    soft_teacher = nn.functional.softmax(t_outputs.squeeze() / temperature, dim=-1)
    soft_student = nn.functional.log_softmax(s_outputs.squeeze() / temperature, dim=-1)

    # Compute distillation loss
    distillation_loss = KLD(soft_student, soft_teacher) * (temperature**2)

    # true label loss
    student_target_loss = s_loss_gn(student_outputs.squeeze(), targets.squeeze())

    # return final distillation loss
    return (1. - lambda_param) * student_target_loss + lambda_param * distillation_loss
    
    
