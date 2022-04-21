# Author: https://github.com/mobecks
# Training script for ensembles of single models
import os
import time
import glob
import pickle
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from torch import nn
from torch.utils.data import Dataset
import ray
from ray import tune


import argparse
parser = argparse.ArgumentParser(description="Run")
parser.add_argument("--raytuning",type=int,default=0) # 0 for standard training
input_args = parser.parse_args()

import sys
MYPATH = ''                         # TODO: insert path to scripts
DATASET_PATH = ''                   # TODO: insert ShimDB path
RAYPATH = ''                        # TODO: insert base path to /ray_results/

sys.path.append(MYPATH+'Utils/')
from models import MyCNNflex_Regr

def seed_everything(seed):
    #random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
SEED = 5121318          # arbitrary number chosen by author.
seed_everything(SEED)

DATAPATH = MYPATH+'/data/'              # should contain rayresult pickle file
if not os.path.exists(DATAPATH):
    os.mkdir(DATAPATH)

RAY_RESULTS_poc_NAS = '/raytune_results_poc.pickle'

initial_config = {
        "set":'poc',                # proof-of-concept
        "base_models": RAY_RESULTS_poc_NAS,
        "train_set": 'large', # 'transfer' dataset or 'large' database
        "nr_models": 50,            # nr weak learners
        "freeze": True,             # freeze weak learners
        "downsample_factor": 16,    # downsample 32768 points by x
        "shift_augm": 10,           # shift z0
        "shift_type": 'complex',     # shift whole input (augmentation)
        "label_noise": 1,         # wrt step_size
        "label_scaling": 100,       # scale to prevent vanishing gradients
        "max_data": 1e5,            # scale to prevent exploding gradients
        "meta_type": 'mlp',         # weak learner architecture
        "filters": 32,
        "LR": 3e-6,
        "MOM": .9,                  # momentum
        "WD": 0,                    # weight decay
        "batch_size": 16,
        "epochs": 50,
        'optimizer': 'SGD',
        "drop_p_ensemble": .1,
    }

global DEVICE
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

#https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-taking-union-of-dictiona
def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

def load_data(pathtodata):
    with open(os.path.dirname(os.path.abspath(DATAPATH))+pathtodata,'rb') as f:
        [data_batched, labels_batched] = pickle.load(f)
    data_batched = data_batched/initial_config["max_data"]
    labels_batched = labels_batched/(2**15)*initial_config["label_scaling"]
    return data_batched, labels_batched

# Create train/val/test sets
def create_sets(data_all, labels_all, seed=52293):
    # !!! prevent leakage
    np.random.seed(seed)

    trainsize, valsize = int(len(data_all)*0.8), int(len(data_all)*0.1)
    set_sizes = [trainsize, valsize, int(len(data_all)-trainsize-valsize)]

    #randomly assign to train set
    rand_idxs = np.random.choice( np.arange(0, len(data_all)), size = set_sizes[0], replace=False)
    data = np.array([ data_all[i] for i in rand_idxs])
    labels = np.array([ labels_all[i] for i in rand_idxs])

    # create val set of remaining
    remaining = list(set(np.arange(0,len(data_all)))  - set(rand_idxs))
    rand_idxs_val = np.random.choice( remaining , size = set_sizes[1], replace=False)
    data_val = np.array([ data_all[i] for i in rand_idxs_val])
    labels_val = np.array([ labels_all[i] for i in rand_idxs_val])

    #create test set of remaining
    remaining = list(set(remaining)  - set(rand_idxs_val))
    rand_idxs_test = np.random.choice( remaining , size = set_sizes[2], replace=False)
    data_test = np.array([ data_all[i] for i in rand_idxs_test])
    labels_test = np.array([ labels_all[i] for i in rand_idxs_test])

    return data, labels, data_val, labels_val, data_test, labels_test

# Dataset class
class MyDataset(Dataset):
    def __init__(self, data, labels, config, transform=False):
        self.max_data = np.max(data)
        self.data = data/self.max_data
        self.labels = labels
        self.transform = transform
        self.ensemble_shift = config["shift_augm"]
        self.config = config
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        if self.transform:
            #heterogen shift
            if self.config["shift_type"] == 'normal':
                batch = torch.roll(torch.tensor(self.data[idx]),
                        np.random.randint(-int(self.ensemble_shift),int(self.ensemble_shift))).float()
            if self.config["shift_type"] == 'complex':
                batch = torch.tensor(np.zeros([4, int(2048)]))
                shifts =np.random.randint(-int(self.ensemble_shift), int(self.ensemble_shift), size=self.data[idx].shape[0])
                for ix, s in enumerate(shifts):
                    batch[ix] = torch.roll(torch.tensor(self.data[idx, ix]), s).float()
            # homogen shift
            if self.config["label_noise"]!=0:
                st = {'poc':1000}[self.config["set"]]
                noise = np.random.randint(-int(self.config["label_noise"]*st),int(self.config["label_noise"]*st),size=3)/32768*self.config["label_scaling"]
            else:
                noise = 0
            return batch, torch.tensor(self.labels[idx]+noise).float()
        return torch.from_numpy(self.data[idx]).float(), torch.tensor(self.labels[idx]).float()


# provide ray_results as string or list of strings
def get_ensemble(config=initial_config):
    sys.path.append(MYPATH+'Utils/')
    from models import MyCNNflex_Regr

    ray_results = config["base_models"]
    nr_models  = config["nr_models"]
    model_list = []

    if len(ray_results) > 2 : ray_results = [ray_results]

    for sub in ray_results:
        with open(DATAPATH + sub, 'rb') as f:
            res = pickle.load(f)
        res_ascending = res.sort_values('loss')
        dirs_topX = res_ascending['logdir'][:nr_models]

        # change paths to your machine
        dirs_topX = [RAYPATH + d for d in dirs_topX]
        
        # create model and load checkpoint (weights) for each weak learner
        for d in dirs_topX:
            with open(d+'\params.json', 'rb') as f:
                config_tmp = json.load(f)

            model = MyCNNflex_Regr(input_shape=(int(config["batch_size"]),4,32768/config["downsample_factor"]),
                                  num_classes = 3, kernel_size=int(config_tmp["kernel_size"]),stride=int(config_tmp["stride"]),
                                  pool_size=int(config_tmp["pool_size"]), num_layers=int(config_tmp["num_layers"]),
                                  drop_p_conv = config_tmp["drop_p_conv"], drop_p_fc = config_tmp["drop_p_fc"])

            last_checkpoint = glob.glob(d+'\checkpoint_*')[-1]
            checkpoint_path = os.path.join(last_checkpoint, "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint_path)
            model.load_state_dict(model_state)
            model.to(DEVICE)
            model.eval()

            if config["freeze"]:
                for param in model.parameters():
                    param.requires_grad_(False)

            model_list.append(model)

    print(config)
    if config['meta_type'] == 'fc': model = MyFCEnsemble(model_list,config)
    if config['meta_type'] == 'linear': model = MyMLPEnsemble(model_list,config)
    if initial_config["meta_type"] == 'average': model = MyAverageEnsemble(model_list)
    return model

# get best weak learner (without ensemble)
def get_single_model(ray_results):
    with open(DATAPATH + ray_results, 'rb') as f:
        res = pickle.load(f)
    res_ascending = res.sort_values('loss')
    dir_top1 = res_ascending['logdir'][:1]
    dir_top1 = dir_top1.iloc[0]
    # change paths in rayresults to match your machine
    dir_top1 = RAYPATH + dir_top1
    print(dir_top1)
    with open(dir_top1+'\params.json', 'rb') as f:
        config = json.load(f)
    model = MyCNNflex_Regr(input_shape=(int(config["batch_size"]),4,32768/initial_config["downsample_factor"]),
                        num_classes = 3, kernel_size=int(config["kernel_size"]),stride=int(config["stride"]),
                        pool_size=int(config["pool_size"]), num_layers=int(config["num_layers"]),
                        drop_p_conv = config["drop_p_conv"], drop_p_fc = config["drop_p_fc"])
    last_checkpoint = glob.glob(dir_top1+'\checkpoint_*')[-1]
    checkpoint_path = os.path.join(last_checkpoint, "checkpoint")
    model_state, optimizer_state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    return model

#%% Ensemble network

# ensemble with multi-layer perceptron (MLP) meta-model
class MyMLPEnsemble(nn.Module):
    def __init__(self, model_list, config, outshape=3):
        super().__init__()
        self.models = model_list
        # Remove last linear layer
        for idx,m in enumerate(self.models):
            features = self.models[idx].fc_block[-1].in_features
            self.models[idx].fc_block[-1] = nn.Identity()
        # Create new classifier
        self.gate = nn.Linear(features*len(self.models), features) # allow non-linear dependecies with ReLU
        self.drop = nn.Dropout(config["drop_p_ensemble"])
        self.regressor = nn.Linear(features, outshape)

    def forward(self, x):
        tmp = torch.Tensor().to(DEVICE)
        for m in self.models:
            tmp = torch.cat((tmp, m(x)), dim=1).to(DEVICE)
        x = self.gate(F.relu(tmp))
        x = self.drop(x)
        x = self.regressor(x)
        return x

# ensemble with fully-connected (FC) meta-model
class MyFCEnsemble(nn.Module):
    def __init__(self, model_list, config, outshape=3):
        super().__init__()
        self.models = model_list
        self.regressor = nn.Linear(3*len(self.models), outshape)
    def forward(self, x):
        tmp = torch.Tensor().to(DEVICE)
        for m in self.models:
            tmp = torch.cat((tmp, m(x)), dim=1).to(DEVICE)
        x = self.regressor(F.relu(tmp))
        return x
        
# ensemble by averaging all weak learners' predictions
class MyAverageEnsemble(nn.Module):
    def __init__(self, model_list, outshape=3):
        super().__init__()
        self.models = model_list
    def forward(self, x):
        tmp = torch.Tensor(x.shape[0], 3).to(DEVICE)
        for m in self.models:
            tmp = (tmp + m(x))
        return tmp / len(self.models)


def train(config, checkpoint_dir=None, num_workers=8, raytune=False):
    sys.path.append(MYPATH+'Utils/')
    from models import MyCNNflex_Regr

    if config["train_set"] == 'transfer':
        data, labels = load_data('/testdataset_firstorder.pickle')
        data = data * initial_config["max_data"]   # undo scaling
    elif config["train_set"] == 'large':
        data, labels = load_data('/preloaded_pickle/batched_data_2048p.pickle')
    data = data * config["max_data"]   # undo scaling
    data, labels, data_val, labels_val, data_test, labels_test = create_sets(data, labels)
    dataset = MyDataset(data, labels, config, transform=(True))
    dataset_val = MyDataset(data_val, labels_val, config, transform=(False))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=config["batch_size"], shuffle=True)

    if initial_config["meta_type"] == 'none_tuned': 
        model = get_single_model(initial_config["base_models"] )
    else: 
        model = get_ensemble(config)
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)

    criterion = nn.MSELoss()
    if config["optimizer"] == 'SGD': optimizer = torch.optim.SGD(model.parameters(), lr=config["LR"],
                                                        momentum=config["MOM"], weight_decay=config["WD"])
    if config["optimizer"] == 'adam': optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"], weight_decay=config["WD"])

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    error_train = []
    error_val = []

    for epoch in range(config["epochs"]):
        if not raytune: print("Epoch #", epoch)
        for mode, dataloader in [("train", train_loader), ("val", val_loader)]:

            if mode == "train":
                model.train()
            else:
                model.eval()

            runningLoss = 0
            total = 0

            for i_batch, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.float(), targets.float()
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)

                runningLoss += loss.item() * inputs.shape[0]
                total += inputs.shape[0]

                if mode == "train":
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            if mode == 'train': error_train.append(runningLoss/total)
            if mode == 'val': error_val.append(runningLoss/total)
        if raytune:
            # Here we save a checkpoint. It is automatically registered with
            # Ray Tune and will potentially be passed as the `checkpoint_dir`
            # parameter in future iterations.
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
            tune.report(loss = error_val[-1], loss_train = error_train[-1]  )

        if not raytune:
            print('Train error: ', round(error_train[-1],4))
            print('Val error: ', round(error_val[-1],4))
            
    return model, error_train, error_val


def test_best_model(best_trial=None, raytune=False, model=None, config=None):
    sys.path.append(MYPATH+'Utils/')
    from models import MyCNNflex_Regr, weights_init

    # allow local and raytune runs.
    if raytune:
        config = best_trial.config
    else:
        config = config

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    if raytune:
        if initial_config["meta_type"] == 'none_tuned': 
            model = get_single_model(initial_config["base_models"] )
        else: 
            model = get_ensemble(config)
            checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint_path)
            model.load_state_dict(model_state)
    model.to(DEVICE)
    model.eval()

    data, labels = load_data('/Models/transfer_set.pickle')
    data = data * config["max_data"]   # undo scaling
    data, labels, data_val, labels_val, data_test, labels_test = create_sets(data, labels)
    dataset_test = MyDataset(data_test, labels_test, config, transform=(False))
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=config["batch_size"], shuffle=True)


    criterion = nn.MSELoss()
    error_test = 0
    total = 0
    
    for i_batch, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.float(), targets.float()
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)

        error_test += loss.item() * inputs.shape[0]
        total += inputs.shape[0]

    if not raytune:
        plt.figure()
        plt.plot(error_train, label = "Train Error")
        plt.plot(error_val, label = "Val Error")
        plt.plot(initial_config["epochs"]-1, error_test/total, 'x', label = "'Test' Error")
        plt.xlabel('Epoch #')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training loss (MSE) for regression NN')
        plt.savefig(DATAPATH + '/loss_MultiRegrNN_{}.png'.format(initial_config["set"]))
        plt.show()

    if initial_config["meta_type"]=='none_tuned': 
        torch.save(model.state_dict(), DATAPATH + '/dre_{}_{}.pt'.format(initial_config["set"], initial_config["meta_type"]))
    else:
        torch.save(model.state_dict(), DATAPATH + '/ensemble_dre_{}_{}.pt'.format(initial_config["set"], initial_config["meta_type"]))

if input_args.raytuning == 0:
    model, error_train, error_val = train(config=initial_config)
    test_best_model(model=model,config=initial_config)


#%% HyperOpt with ray tune
from ray.tune.schedulers import ASHAScheduler   # early stopping
from ray.tune.suggest.basic_variant import BasicVariantGenerator

def main(num_samples=10, max_num_epochs=2000, gpus_per_trial=.5, num_workers=8):
    # search space

    config = {
        "base_models": RAY_RESULTS_poc_NAS,
        "train_set": tune.choice(['large', 'transfer']),
        "nr_models": tune.choice([10,50]),
        "shift_augm": tune.choice([5,10,100]),
        "freeze": tune.choice([False, True]),           # ignore freeze False since model loading will not be possible on Magritek PC! 
        "shift_type": tune.choice(['normal', 'complex']),
        "label_noise": tune.choice([0,0.1,0.5,1]),      # wrt step_size
        "meta_type": tune.choice(['linear', 'fc']),
        "batch_size": tune.choice([8,16,32,64,128]),
        "LR": tune.loguniform(1e-6, 1e-4, 5e-4),
        "MOM": .9,
        "WD": tune.choice([0,1e-4,1e-2,0.1,0.5,1,2,10]),
        "epochs": tune.choice([50,100]),
        'optimizer': tune.choice(['adam','SGD']),
        "drop_p_ensemble": tune.choice([0,.1,0.2,0.5,0.8])
    }

    config = merge_two_dicts(initial_config, config)   #merge configsS

    # early stoppping scheduler
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=20, # min. nr of trained epochs
        reduction_factor=2)

    # random search
    searcher =BasicVariantGenerator()

    result = tune.run(
        tune.with_parameters(train, num_workers=num_workers,raytune=True),
        resources_per_trial={"gpu": gpus_per_trial},
        config=config,
        metric="loss",
        mode="min",
        name="raytune_ensemble_{}".format(initial_config['set']),
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=searcher,
        raise_on_failed_trial  = False, # allow errors
        keep_checkpoints_num = 1, # reduce disc load
        checkpoint_score_attr = 'min-loss',
    )

    best_trial = result.get_best_trial("loss", "min", "all")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    if ray.util.client.ray.is_connected():
        # If using Ray Client, we want to make sure checkpoint access
        # happens on the server. So we wrap `test_best_model` in a Ray task.
        # We have to make sure it gets executed on the same node that
        # ``tune.run`` is called on.
        from ray.tune.utils.util import force_on_current_node
        remote_fn = force_on_current_node(ray.remote(tune.with_parameters(test_best_model, raytune=True)))
        ray.get(remote_fn.remote(best_trial))
    else:
        test_best_model(best_trial, raytune=True)

    return result

if input_args.raytuning == 1:
    result = main(num_samples=500,max_num_epochs=initial_config["epochs"], gpus_per_trial=.2)
    df = result.dataframe()

    with open(DATAPATH + '/raytune_ensemble_results_{}_{}.pickle'.format(initial_config["set"],datetime.today().strftime('%Y-%m-%d')), 'wb') as f:
        pickle.dump(df, f)
