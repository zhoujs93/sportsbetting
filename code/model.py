import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.optim import swa_utils
from transformers import AdamW, get_cosine_schedule_with_warmup

class FeedForwardNN(nn.Module):

    def __init__(self, emb_dims, no_of_cont, lin_layer_sizes, output_size, emb_dropout, lin_layer_dropouts, use_embedding):

        super().__init__()
        self.use_embedding = use_embedding
        self.emb_dims = emb_dims
        if self.use_embedding:
        # Embedding layers
            self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                             for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])

        self.no_of_embs = no_of_embs

        self.no_of_cont = no_of_cont
        self.lin_layer_sizes = lin_layer_sizes
        self.output_size = output_size
        self.emb_dropout = emb_dropout
        self.lin_layer_dropouts = lin_layer_dropouts
        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont,
                                    lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList([first_lin_layer] + [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
                                                             for i in range(len(lin_layer_sizes) - 1)])

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1],
                                      output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(lin_layer_sizes[i]) for i in range(len(lin_layer_sizes))])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])

    def forward(self, cat_data, cont_data):

        if self.no_of_embs != 0:
            if self.use_embedding:
                x = [emb_layer(cat_data[:,i])
                     for i, emb_layer in enumerate(self.emb_layers)]
                x = torch.cat(x, dim = 1)
                x = self.emb_dropout_layer(x)

            else:
                x = cat_data

        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)

            if self.no_of_embs != 0:
                x = torch.cat([x, normalized_cont_data], 1)
            else:
                x = normalized_cont_data

        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self.droput_layers, self.bn_layers):
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)

        return x

    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location= lambda storage, loc: storage)
        args = params['args']
        model = FeedForwardNN(**args)
        model.load_state_dict(params['state_dict'])

    def save(self, path: str):
        print(f'Save model parametrs to {path}')
        params = {
            'args' : dict(emb_dims = self.emb_dims,
                          no_of_cont = self.no_of_cont,
                          lin_layer_sizes = self.lin_layers,
                          output_size = self.output_size,
                          emb_dropout = self.emb_dropout,
                          lin_layer_dropouts = self.lin_layer_dropouts,
                          use_embedding = self.use_embedding),
            'state_dict' : self.state_dict()
        }
        torch.save(params, path)

device = torch.device("cuda:0")

class dataset(Dataset):
    def __init__(self, categorical, numerical, data, labels):
        self.labels = labels
        self.data = data
        self.categorical_features = categorical
        self.numerical_features = numerical

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        categorical_data = self.data[self.categorical_features].iloc[index].values
        numerical_data = self.data[self.numerical_features].iloc[index].values
        label = self.labels[index]
        categorical_data_tensor = torch.tensor(categorical_data, dtype = torch.long)
        numerical_data_tensor = torch.tensor(numerical_data, dtype = torch.float)
        label = torch.tensor(label, dtype = torch.float).unsqueeze(0)
        return (categorical_data_tensor, numerical_data_tensor, label)

def accuracy(ytrue, preds, covered = False):
    if covered:
        mask = preds != 2
        ytrue = ytrue[mask]
        preds = preds[mask]

    scores = (preds.long() == ytrue).sum().item()
    return scores / ytrue.size(0)

def eval_model(valid_dataloader, model, loss_fn, args):
    was_training = model.training
    model.eval()
    valid_loss = []
    acc = []
    with torch.no_grad():
        for valid_categorical, valid_numerical, valid_label in valid_dataloader:
            valid_categorical = valid_categorical.to(device)
            valid_numerical = valid_numerical.to(device)
            valid_label = valid_label.to(device)
            log_probs_valid = model(valid_categorical, valid_numerical)
            loss_valid = loss_fn(log_probs_valid, valid_label)
            # if args['objective'] == 'classification':
            if args['objective'] == 'regression':
                acc.append(loss_valid.item())
            else:
                # valid_predictions = torch.sigmoid(log_probs_valid)
                valid_predictions = torch.nn.functional.softmax(log_probs_valid)
                valid_binary_predictions = torch.argmax(valid_predictions, axis = 1)
                if args['loss_fn'] == 'gambler_loss':
                    valid_acc_score = accuracy(valid_label, valid_binary_predictions, covered = True)
                else:
                    valid_acc_score = accuracy(valid_label, valid_binary_predictions)
                acc.append(valid_acc_score)
            # else:
            #     valid_binary_predictions = torch.argmax(log_probs_valid, dim = 1)
            #     valid_acc_score = accuracy(valid_label, valid_binary_predictions)
            #     acc.append(valid_acc_score)

            valid_loss.append(loss_valid.item())

    if was_training:
        model.train()
    # if args['objective'] == 'classification':
    return np.mean(valid_loss), np.mean(acc)

def gambler_loss(model_output, targets, reward = 1.9):
    outputs = torch.nn.functional.softmax(model_output, dim = 1)
    outputs, reservation = outputs[:, :-1], outputs[:, -1]
    gain = torch.gather(outputs, dim = 1, index = targets.unsqueeze(1)).squeeze()
    doubling_rate = (gain + reservation/reward).log()
    return -doubling_rate.mean()

def make_predictions(test_dataloader, model, args):
    was_training = model.training
    model.eval()
    predictions = []
    losses = []
    # loss = torch.nn.BCEWithLogitsLoss()
    loss = torch.nn.CrossEntropyLoss()
    acc = []
    preds = []
    with torch.no_grad():
        for test_categorical, test_numerical, test_label in test_dataloader:
            test_categorical = test_categorical.to(device)
            test_numerical = test_numerical.to(device)
            test_label = test_label.to(device)
            log_probs_test = model(test_categorical, test_numerical)
            loss_test = loss(log_probs_test, test_label)
            preds.append(log_probs_test.cpu())
            # if args['objective'] == 'classification':
            if args['objective'] == 'regression':
                predictions.append(log_probs_test.cpu().data.numpy())
            else:
                # test_predictions = torch.sigmoid(log_probs_test)
                test_predictions = torch.nn.functional.softmax(log_probs_test)
                test_binary_predictions = torch.argmax(test_predictions, dim = 1)
                # if args['loss_fn'] == 'gambler_loss':
                #     test_acc_score = accuracy(test_label, test_binary_predictions, covered = True)
                # else:
                #     test_acc_score = accuracy(test_label, test_binary_predictions)
                # acc.append(test_acc_score)
                # losses.append(loss_test.item())
                predictions.append(test_predictions.cpu().data.numpy())
            # else:
            #     predictions.append(log_probs_test.cpu().data.numpy())
    if was_training:
        model.train()
    return np.vstack(predictions), torch.cat(preds, dim = 0)


def train(args, patience = 25, train_dataset = None, valid_dataset = None, early_stopping = True,
          num_epochs = None, dataset = None, pos_weights = None):
    device = torch.device("cuda:0")

    emb_dims = args['emb_dims']
    continuous_features = args['continuous_features']
    lin_layer_sizes = args['lin_layer_sizes']
    if args['loss_fn'] == 'gambler_loss':
        output_size = args['output_size'] + 1
    else:
        output_size = args['output_size']
    embedding_dropout = args['embedding_dropout']
    lin_layer_dropouts = args['linear_layer_dropout']
    lr = args['lr']
    use_embedding = args['use_embedding']
    log_dir = args['log_dir']
    # objective = args['objective']
    if 'weight_decay' in args.keys():
        weight_decay = args['weight_decay']
    else: weight_decay = 0

    # torch.manual_seed(args['seed'])
    # np.random.seed(args['seed'])

    model = FeedForwardNN(emb_dims, continuous_features,
                          lin_layer_sizes, output_size,
                          embedding_dropout, lin_layer_dropouts, use_embedding=use_embedding)

    # if objective == 'classification':
    if args['objective'] == 'regression':
        # loss_fn = torch.nn.MSELoss(reduction='mean')
        loss_fn = torch.nn.SmoothL1Loss(reduction = 'mean')
    elif args['loss_fn'] == 'gambler_loss':
        loss_fn = gambler_loss
    else:
        if pos_weights is not None:
            weights = torch.tensor([pos_weights]).cuda()
            # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = weights)
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
    # else:
    #     loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = float(lr), weight_decay=weight_decay)

    if args['swa']:
        swa_model = swa_utils.AveragedModel(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    prev_valid_loss, counter, optimal_num_epochs, epochs = np.inf, 0, 1, 0
    epoch_for_cosine = 20
    writer = SummaryWriter(log_dir = log_dir)
    if early_stopping:
        if args['swa']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
            swa_start = 100
            swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr = 1e-4)
        else:
            # scheduler = CosineAnnealingLR(optimizer, T_max = 100, eta_min = 0.1)
            # scheduler = StepLR(optimizer, step_size=20, gamma = 0.5)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
            # scheduler = ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 20, verbose=True)
            if args['scheduler'] == 'cosine':
                scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=3000,
                                                            num_training_steps=epoch_for_cosine * len(train_dataset))
            else:
                scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
            print(scheduler)
        while counter < patience:
            epochs += 1
            losses, train_acc_list = [], []
            train_acc = 0
            for train_categorical, train_numerical, train_label in train_dataset:
                train_categorical = train_categorical.to(device)
                train_numerical = train_numerical.to(device)
                train_label = train_label.to(device)
                optimizer.zero_grad()
                log_probs = model(train_categorical, train_numerical)

                loss = loss_fn(log_probs, train_label)

                loss.backward()

                losses.append(loss.item())

                optimizer.step()

                # if args['objective'] == 'classification':
                if args['objective'] == 'regression':
                    train_acc_list.append(loss.item())
                else:
                    train_prediction = torch.nn.functional.softmax(log_probs)
                    # train_binary_prediction = (train_prediction > 0.5)
                    train_binary_prediction = torch.argmax(train_prediction, dim = 1)
                    if args['loss_fn'] == 'gambler_loss':
                        train_acc = accuracy(train_label, train_binary_prediction, covered = True)
                    else:
                        train_acc = accuracy(train_label, train_binary_prediction)
                    train_acc_list.append(train_acc)
                # else:
                #     train_binary_prediction = torch.argmax(train_prediction, dim = 1)
                #     train_acc = accuracy(train_label, train_binary_prediction)
                #     train_acc_list.append(train_acc)
                if args['swa']:
                    if counter > swa_start:
                        swa_model.update_parameters(model)
                        swa_scheduler.step()
                    else:
                        scheduler.step()
                else:
                    scheduler.step()

            if args['swa']:
                valid_loss, acc_score = eval_model(valid_dataset, swa_model, loss_fn, args)
            else:
                valid_loss, acc_score = eval_model(valid_dataset, model, loss_fn, args)

            if args['objective'] == 'regression':
                tr_info = {'loss' : np.mean(losses), 'train_mse' : np.mean(train_acc_list).item()}
                valid_info = {'valid_loss' : valid_loss, 'valid_mse' : acc_score}
            else:
                tr_info = {'loss' : np.mean(losses), 'train_accuracy' : np.mean(train_acc_list).item()}
                valid_info = {'valid_loss' : valid_loss, 'valid_accuracy' : acc_score}

            for tag, value in tr_info.items():
                writer.add_scalar(tag,value, epochs)

            for val_tag, val_value in valid_info.items():
                writer.add_scalar(val_tag, val_value, epochs)

            if args['scheduler'] != 'cosine':
                scheduler.step(valid_loss.item())
            if args['objective'] == 'regression':
                print('Epoch %i : Train Loss of %0.5f ; Valid Loss of %0.5f ; MSE of %0.5f'
                      %(epochs, np.mean(losses).item(), valid_loss.item(), acc_score.item()))
            else:
                print('Epoch %i : Train Loss of %0.5f ; Valid Loss of %0.5f ; Accuracy of %0.5f'
                      %(epochs, np.mean(losses).item(), valid_loss.item(), acc_score.item()))


            if valid_loss.item() < prev_valid_loss:
                counter = 0
                optimal_num_epochs = epochs
                prev_valid_loss = valid_loss.item()
            else:
                counter += 1
        if args['swa']:
            swa_utils.update_bn(train_dataset, swa_model)
        print('Early Stopping, Best Optimal Number of Epoch is %i' %(optimal_num_epochs))
        return optimal_num_epochs
    else:
        if args['swa']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
            swa_start = 100
            swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr= 1e-4)
        else:
            # scheduler = CosineAnnealingLR(optimizer, T_max = 100, eta_min = 0.1)
            # scheduler = StepLR(optimizer, step_size=20, gamma = 0.5)
            epoch_for_cosine = 20
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=3000,
                                                        num_training_steps=epoch_for_cosine * len(dataset))
            print(scheduler)
            # scheduler = ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 20, verbose=True)
        # scheduler = CosineAnnealingLR(optimizer, T_max = 100, eta_min = 0.1)
        print('Training For Optimal Number of Epochs %i Based on Early Stopping' %(num_epochs))
        for epoch in range(num_epochs):
            model.train()
            losses = []
            for categorical, numerical, label in dataset:
                categorical = categorical.to(device)
                numerical = numerical.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                log_probs = model(categorical, numerical)

                loss = loss_fn(log_probs, label)

                loss.backward()

                losses.append(loss.item())

                optimizer.step()

                if args['swa']:
                    if counter > swa_start:
                        swa_model.update_parameters(model)
                        swa_scheduler.step()
                    else:
                        scheduler.step()
                else:
                    scheduler.step()

            print('Epoch %i : Entire Train Loss of %0.5f' %(epoch, np.mean(losses).item()))
        if args['swa']:
            swa_utils.update_bn(train_dataset, swa_model)
            return swa_model, np.mean(losses).item()

        return model, np.mean(losses).item()