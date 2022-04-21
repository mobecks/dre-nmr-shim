# Author: https://github.com/mobecks
# Training script for single models
import os
import time
import pickle
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from torch import nn
from torch.utils.data import Dataset
import ray
from ray import tune

# install dev version: python -m pip install git+https://github.com/jjhelmus/nmrglue.git@6ca36de7af1a2cf109f40bf5afe9c1ce73c9dcdc

import argparse
parser = argparse.ArgumentParser(description="Run")
parser.add_argument("--raytuning",type=int,default=0) # 0 for standard training, 1 for raytuning
input_args = parser.parse_args()

import sys
MYPATH = ''                         #TODO: insert path to scripts
DATASET_PATH = ''                   #TODO: insert path to ShimDB
sys.path.append(MYPATH+'Utils/')    # import own util functions

DATAPATH = MYPATH+'/data/'
if not os.path.exists(DATAPATH):
    os.mkdir(DATAPATH)

initial_config = {
        "set":'poc',                # proof-of-concept
        "downsample_factor": 16,    # downsample 32768 points by x
        "shift_augm": 256,          # shift z0
        "shift_type": 'normal',     # shift whole input
        "label_noise": 0,           # wrt step_size
        "label_scaling": 100,       # scale to prevent vanishing gradients
        "max_data": 1e5,            # scale to prevent exploding gradients
        "num_layers": 5,            # Nr. layers in CNN
        "kernel_size": 51,
        "stride": 2,
        "pool_size": 1,
        "filters": 32,
        "LR": 1e-3,
        "batch_size": 32,
        "epochs": 150,
        'optimizer': 'adam',
        "enable_subbatches": False, # Use if GPU is small
        "drop_p_conv":.2,
        "drop_p_fc":.1
    }

#if GPU runs out of memory consider changing the batch size

#https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20
#if BATCH_SIZE is to big for GPU, enable sub-batches by setting TRUE

if initial_config["enable_subbatches"] == True:
    DIVISOR = 8 #has to be divider of inital batch size
    BATCH_SIZE = int(initial_config["batch_size"] / DIVISOR)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
SEED = 5
seed_everything(SEED)


#%% Creating batched input

def load_data():
    # load whole pickle package from disc. Save iteration time.
    # '/preloaded_pickle/batched_data_2048p.pickle' = pickle file created with Prepickle_Data.py
    with open(os.path.dirname(os.path.abspath(DATASET_PATH))+'/preloaded_pickle/batched_data_2048p.pickle','rb') as f:
        [data_tmp, labels_tmp, dic] = pickle.load(f)

    if initial_config["set"] == 'poc':
        data_batched = data_tmp[0]
        labels_batched = labels_tmp[0]

    data_batched = data_batched/initial_config["max_data"]
    labels_batched = labels_batched/(2**15)*initial_config["label_scaling"]

    return data_batched, labels_batched

#%% train/val/test splitting

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

#%% Dataset class

class MyDataset(Dataset):
    def __init__(self, data, labels, config, transform=False):
        self.data = data
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
                noise = np.random.randint(-int(self.config["label_noise"]*st),
                                          int(self.config["label_noise"]*st),size=3)/32768*self.config["label_scaling"]
            else:
                noise = 0
            return batch, torch.tensor(self.labels[idx]+noise).float()
        return torch.from_numpy(self.data[idx]).float(), torch.tensor(self.labels[idx]).float()

#%% training

def train(config, checkpoint_dir=None, num_workers=8, raytune=False):
    sys.path.append(MYPATH+'Utils/')                    # re-import for raytune nodes
    from models import MyCNNflex_Regr

    model = MyCNNflex_Regr(input_shape=(int(config["batch_size"]),4,32768/initial_config["downsample_factor"]),
                          num_classes = 3, kernel_size=int(config["kernel_size"]),stride=int(config["stride"]),
                          pool_size=int(config["pool_size"]), num_layers=int(config["num_layers"]),
                          drop_p_conv = config["drop_p_conv"], drop_p_fc = config["drop_p_fc"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)  # allow multi GPU training
    model.to(device)

    # data loading and wrapping into dataloader
    data_all, labels_all = load_data()
    data, labels, data_val, labels_val, data_test, labels_test = create_sets(data_all, labels_all)
    dataset = MyDataset(data, labels, config, True)
    dataset_val = MyDataset(data_val, labels_val, config)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=int(config["batch_size"]), shuffle=True)

    criterion = nn.MSELoss()
    if config["optimizer"]=='adam':optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"])

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    error_train = []
    error_val = []

    # loop epochs
    for epoch in range(initial_config["epochs"]):
        if not raytune: print("Epoch #", epoch)
        start_t = time.time()
        for mode, dataloader in [("train", train_loader), ("val", val_loader)]:

            if mode == "train":
                model.train()
            else:
                model.eval()
                state = model.state_dict()
                state_error = error_val[-1] if error_val else 0

            runningLoss = 0
            total = 0
            
            # loop batches
            for i_batch, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.float(), targets.float()
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)

                runningLoss += loss.item() * inputs.shape[0]
                total += inputs.shape[0]

                if mode == "train":
                    loss.backward()
                    if initial_config["enable_subbatches"] == True:
                        if (i_batch+1)%DIVISOR == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()

                # delete variables to reduce hardware load
                del inputs
                del targets
                del outputs
                torch.cuda.empty_cache()

            (error_train if mode == "train" else error_val).append(runningLoss / total)

        if raytune:
            # Here we save a checkpoint. It is automatically registered with
            # Ray Tune and will potentially be passed as the `checkpoint_dir`
            # parameter in future iterations.
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
            tune.report(loss = error_val[-1], loss_train = error_train[-1]  )


        end_t = time.time()
        if not raytune:
            print('Train error: ', round(error_train[-1],4))
            print('Val error: ', round(error_val[-1],4))
            print('Time epoch: ', round(end_t - start_t))

    return model, error_train, error_val


# %% Testing

def test_best_model(best_trial=None, raytune=False, model=None, config=None):
    sys.path.append(MYPATH+'Utils/')
    from models import MyCNNflex_Regr

    # allow local and raytune runs.
    if raytune:
        config = best_trial.config
    else:
        config = config

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if raytune:
        model = MyCNNflex_Regr(input_shape=(int(config["batch_size"]),4,32768/initial_config["downsample_factor"]),
                          num_classes = 3, kernel_size=int(config["kernel_size"]),stride=int(config["stride"]),
                          pool_size=int(config["pool_size"]), num_layers=int(config["num_layers"]),
                          drop_p_conv = config["drop_p_conv"], drop_p_fc = config["drop_p_fc"])
        checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint_path)
        model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    data_all, labels_all = load_data()
    data, labels, data_val, labels_val, data_test, labels_test = create_sets(data_all, labels_all)
    dataset_test = MyDataset(data_test, labels_test, config)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=int(config["batch_size"]), shuffle=True)

    criterion = nn.MSELoss()
    error_test = 0
    total = 0
    mae = np.array([])

    for i_batch, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.float(), targets.float()
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)

        error_test += loss.item() * inputs.shape[0]
        total += inputs.shape[0]

        mae = np.append(mae, abs(outputs.cpu().detach().numpy())-abs(targets.cpu().detach().numpy()))

    mae = np.reshape(mae, (-1,3) )

    print('\n NN: MAE test set {} +/- {}'.format(np.mean(np.abs(mae)), np.std(np.abs(mae))))

    if not raytune:
        plt.figure()
        plt.plot(error_train, label = "Train Error")
        plt.plot(error_val, label = "Val Error")
        plt.plot(initial_config["epochs"]-1, error_test/total, 'x', label = "'Test' Error")
        plt.xlabel('Epoch #')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training loss (MSE) for DRE')
        plt.savefig(DATAPATH + '/loss_dre_{}.png'.format(initial_config["set"]))
        plt.show()

    torch.save(model.state_dict(), DATAPATH + '/model_dre_{}.pt'.format(initial_config["set"]))

# run train and test without raytuning if not input_args.raytuning
if input_args.raytuning == 0:
    model, error_train, error_val = train(config=initial_config)
    test_best_model(model=model,config=initial_config)


#%% HyperOpt with ray tune
from ray.tune.schedulers import ASHAScheduler   # early stopping
from ray.tune.suggest.basic_variant import BasicVariantGenerator


def main(num_samples=10, max_num_epochs=150, gpus_per_trial=.5, num_workers=8):
    # search space; chosen arbitrary by author.
    config = {
        "num_layers": tune.choice([3,4,5]),
        "kernel_size": tune.choice([11,21,31,41,51,71]),
        "stride": tune.choice([1,2,4]),
        "pool_size": tune.choice([1,2]),
        "filters": tune.choice([32,64]),
        "LR": 1e-3,
        "batch_size": 32,
        'optimizer': tune.choice(['adam']),
        "shift_augm": 256,
        "drop_p_conv": .2,
        "drop_p_fc": .5
    }

    # early stoppping scheduler
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=5, # min. nr of trained epochs
        reduction_factor=2)
    # random search over search space
    searcher =BasicVariantGenerator(points_to_evaluate=[{"kernel_size": 19, "stride": 8, "num_layers": 3, "pool_size":1},
                        {"num_layers": 5, "kernel_size":11, "stride":2,"pool_size":1, "filters":32},
                        {"kernel_size": 41, "stride": 1}])

    result = tune.run(
        tune.with_parameters(train, num_workers=8,raytune=True),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        metric="loss",
        mode="min",
        name="raytune_{}".format(initial_config['set']),
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=searcher,
        raise_on_failed_trial  = False, # allow errors
        keep_checkpoints_num = 3, # reduce disc load
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
    result = main(num_samples=300,max_num_epochs=initial_config["epochs"], gpus_per_trial=.2)
    df = result.dataframe()

    with open(DATAPATH + '/raytune_results_{}_{}.pickle'.format(initial_config["set"],datetime.today().strftime('%Y-%m-%d')), 'wb') as f:
        pickle.dump(df, f)


# investigate results e.g. in spyder
# =============================================================================
# fn = '/raytune_results_poc.pickle' #filename of result
# with open(DATAPATH + fn, 'rb') as f:
#     res = pickle.load(f)
# =============================================================================
