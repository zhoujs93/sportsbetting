import ray
from ray import tune
import ray.tune.schedulers
ray.init()
import pandas as pd
import model
import pickle
import numpy as np
import feather
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from ray.tune.analysis import ExperimentAnalysis

def tune_nn(args, config, reporter):
    args = config.pop('params')
    args.update(config)
    args["linear_layer_dropout"] = [args["linear_layer_dropout"]] * len(args["nn_layers"])

    X_train_df = feather.read_dataframe(args['dir'] + 'data/X_train_df.feather')
    y_train_df = feather.read_dataframe(args['dir'] + 'data/y_train_df.feather')
    X_valid_df = feather.read_dataframe(args['dir'] + 'data/X_valid_df.feather')
    y_valid_df = feather.read_dataframe(args['dir'] + 'data/y_valid_df.feather')
    X_test_df = feather.read_dataframe(args['dir'] + 'data/X_test_df.feather')
    y_test_df = feather.read_dataframe(args['dir'] + 'data/y_test_df.feather')

    with open(args['dir'] + 'data/features.pkl', 'rb') as f:
        features = pickle.load(f)
    with open(args['dir'] + 'data/categorical_features.pkl', 'rb') as f:
        categorical_features = pickle.load(f)
    with open(args['dir'] + 'data/cat_sizes.pkl', 'rb') as f:
        cat_szs = pickle.load(f)

    args['emb_dims'] = cat_szs
    args['continuous_features'] = len(features)

    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])


    cont_train_tensor = torch.from_numpy(X_train_df[features].values).float()
    cat_train_tensor = torch.from_numpy(X_train_df[categorical_features].values.astype('int')).long()

    cont_valid_tensor = torch.from_numpy(X_valid_df[features].values).float()
    cat_valid_tensor = torch.from_numpy(X_valid_df[categorical_features].values.astype('int')).long()

    train_label_tensor = torch.from_numpy(y_train_df.values.astype('float')).float().view(-1,1)
    valid_label_tensor = torch.from_numpy(y_valid_df.values.astype('float')).float().view(-1,1)

    print(train_label_tensor.size())

    train_dataset = torch.utils.data.TensorDataset(cat_train_tensor,
                                             cont_train_tensor,
                                             train_label_tensor)

    valid_dataset = torch.utils.data.TensorDataset(cat_valid_tensor,
                                                   cont_valid_tensor,
                                                   valid_label_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 256, shuffle = True, drop_last=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle = True,
                                                   drop_last= True, pin_memory=True)

    epochs = model.train(args, patience = 25, train_dataset = train_loader, valid_dataset = valid_loader)

    dataset = torch.utils.data.ConcatDataset([train_dataset, valid_dataset])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle = True, drop_last=True, pin_memory=True)

    nn_model, loss = model.train(args, patience = 25, num_epochs = epochs, early_stopping = False, dataset = dataloader)

    cont_test_tensor = torch.from_numpy(X_test_df[features].values).float()
    cat_test_tensor = torch.from_numpy(X_test_df[categorical_features].values.astype('int')).long()
    test_label_tensor = torch.from_numpy(y_test_df.values.astype('float')).float().view(-1,1)

    test_dataset = torch.utils.data.TensorDataset(cat_test_tensor,
                                                  cont_test_tensor,
                                                  test_label_tensor)

    test_loader= torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle = False, drop_last=False,
                                             pin_memory=True)

    train_pred_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle = False, drop_last=False, pin_memory=True)
    valid_pred_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=False, drop_last=False, pin_memory=True)

    y_test_pred, log_preds_test = model.make_predictions(test_loader, nn_model)
    y_train_pred, log_preds_train = model.make_predictions(train_pred_loader, nn_model)
    y_valid_pred, log_preds_valid = model.make_predictions(valid_pred_loader, nn_model)

    loss = torch.nn.BCEWithLogitsLoss()
    valid_loss = loss(log_preds_valid, valid_label_tensor)
    test_loss = loss(log_preds_test, test_label_tensor)
    train_loss = loss(log_preds_train, train_label_tensor)

    y_pred_binary = (y_test_pred > 0.5)
    y_pred_train_binary = (y_train_pred > 0.5)
    y_pred_valid_binary = (y_valid_pred > 0.5)

    accuracy = accuracy_score(y_test_df.values, y_pred_binary)
    valid_accuracy = accuracy_score(y_valid_df.values, y_pred_valid_binary)
    train_accuracy = accuracy_score(y_train_df.values, y_pred_train_binary)


    reporter(test_mean_loss=test_loss.item(), test_mean_accuracy=accuracy,
             valid_mean_loss = valid_loss.item(), valid_mean_accuracy = valid_accuracy,
             train_mean_loss = train_loss.item(), train_mean_accuracy = train_accuracy)

    print('Train Accuracy is {}'.format(accuracy_score(y_train_df.values, y_pred_train_binary)))
    print('Valid Accuracy is {}'.format(accuracy_score(y_valid_df.values, y_pred_valid_binary)))
    print('Test Accuracy is {}'.format(accuracy_score(y_test_df.values, y_pred_binary)))

if __name__ == '__main__':
    smoke_test = True

    args = {
        'lin_layer_sizes': [512, 256],
        'output_size': 1,
        'embedding_dropout': 0.2,
        'linear_layer_dropout': [0.5, 0.5],
        'seed' : 232425,
        'dir' : '/media/johnz/T Drive1/Google Drive/Stanford SCPD/CS221/project/'
    }

    sched = tune.schedulers.AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="test_mean_accuracy"
    )

    tune.register_trainable(
        "TRAIN_FN",
        lambda config, reporter: tune_nn(args, config, reporter))

    tune.run(
        "TRAIN_FN",
        name="nn_3layers",
        scheduler=sched,
        local_dir="/media/johnz/T Drive1/Google Drive/Stanford SCPD/CS221/project/output",
        **{
            "stop": {
                "test_mean_accuracy": 0.98,
                "training_iteration": 1 if smoke_test else 20
            },
            "resources_per_trial": {
                "cpu": 1,
                "gpu": 0.2
            },
            "num_samples": 1 if smoke_test else 20,
            "config": {
                "lr": tune.sample_from(lambda x: 10**(np.random.uniform(-4,-1))),
                "linear_layer_dropout" : tune.uniform(0.05,0.95),
                "params" : args,
                "nn_layers": [
                    tune.grid_search([16, 64, 256, 512]),
                    tune.grid_search([16, 64, 256, 512]),
                    tune.grid_search([16, 64, 256, 512])
                ],
            },
        })
    #
    ea = ExperimentAnalysis(experiment_path="/media/johnz/T Drive1/Google Drive/Stanford SCPD/CS221/project/output/nn_step_LR")
    trials_dataframe = ea.dataframe()