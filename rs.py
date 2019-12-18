#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import logging
import sagemaker_containers
import requests
import pandas as pd
import numpy as np

import os
import io
import glob
import time
import torch

import numpy as np
import torch
import six
from six import StringIO, BytesIO
import fastai

from fastai.tabular import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# set the constants for the content types
JSON_CONTENT_TYPE = 'application/json'

        
def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Device Type: {}".format(device))

    logger.info("Loading Pets dataset")
    print(f'Batch size: {args.batch_size}')
    path = Path(args.data_dir)
    print(f'Data path is: {path}')


    
    # get the pattern to select the training/validation data
    print('Creating DataBunch object')
    df = pd.read_csv(path/'adult.csv')
    dep_var = 'salary'
    cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
    cont_names = ['age', 'fnlwgt', 'education-num']
    procs = [FillMissing, Categorify, Normalize]
    test = TabularList.from_df(df.iloc[800:1000].copy(), path=path, cat_names=cat_names, cont_names=cont_names)

    data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(800,1000)))
                           .label_from_df(cols=dep_var)
                           .add_test(test)
                           .databunch())

    # create the tabular model

    print("Creating neural net")    
    learn = tabular_learner(data, layers=[200,100], metrics=accuracy)

    print('Fit for 4 cycles')
    #fit for two on trial
    learn.fit_one_cycle(1)    
    learn.unfreeze()
    print('Unfreeze and fit for another 2 cycles')
    #fit for 1 on trial

    #learn.fit_one_cycle(1, max_lr=slice(1e-6,1e-4))
    print('Finished Training')
    
    logger.info("Saving the model.")
    model_path = Path(args.model_dir)

    print(f'Export data object')
    data.export(model_path/'export.pkl')

    # create empty models dir
    os.mkdir(model_path/'models')
    print(f'Saving model weights')
    return learn.save(model_path)

def model_fn(model_dir):
    logger.info('model_fn')
    path = Path(model_dir)
    empty_data = TabularList.load_empty(path)
    learn = tabular_learner(empty_data, layers=[200,100], metrics=accuracy)
    return learn


def input_fn(request_body, request_content_type):
    from predict_fn
    """An input_fn that loads a pickled numpy array"""
    # print("request_body=",str(request_body))
    # print("np.load(StringIO(request_body))=",np.load(StringIO(request_body)))

    if request_content_type == "application/python-pickle":
        array = np.load(BytesIO((request_body)))
        # print("array=",array)
        return array
    elif request_content_type == 'application/json':
        jsondata = json.load(StringIO(request_body))
        normalized_data, benchmark_data = process_input_data(jsondata)
        # print("normalized_data=",normalized_data)
        return normalized_data, benchmark_data
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        raise ValueError("{} not supported by script!".format(request_content_type))




def predict_fn(input_data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        return model(input_data.to(device))


def output_fn(prediction, accept):
    """Format prediction output
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        return worker.Response(json.dumps(prediction), accept, mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), accept, mimetype=accept)
    else:
        raise ValueError("{} accept type is not supported by this script.".format(accept))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS',
                        help='batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--dist_backend', type=str, default='gloo', help='distributed backend (default: gloo)')
    
    env = sagemaker_containers.training_env()
    
    parser.add_argument('--hosts', type=list, default=env.hosts)
    parser.add_argument('--current-host', type=str, default=env.current_host)
    parser.add_argument('--model-dir', type=str, default=env.model_dir)
    parser.add_argument('--data-dir', type=str, default=env.channel_input_dirs.get('training'))
    parser.add_argument('--num-gpus', type=int, default=env.num_gpus)

    _train(parser.parse_args())
    

