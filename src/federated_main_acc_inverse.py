#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
import datetime
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights_acc_inverse, exp_details


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    global_epoch_acc = []
    global_epoch_loss = []
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses, local_acc_list = [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss, model_ = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_acc, _ = local_model.inference(model=model_)
            if local_acc > 0.001:
                local_acc_list.append(local_acc)
            else:
                local_acc_list.append(0.001)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights_acc_inverse(local_weights,local_acc_list)


        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            #local_acc, local_loss = local_model.inference(model=local_model)
            #list_local_acc.append(local_acc)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
        global_epoch_acc.append(100*train_accuracy[-1])
        global_epoch_loss.append(np.mean(np.array(train_loss)))
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    train_accuracy_number = 100*train_accuracy[-1]
    test_acc_number = 100*test_acc
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(train_accuracy_number))
    print("|---- Test Accuracy: {:.2f}%".format(test_acc_number))

    # Saving the objects train_loss and train_accuracy:
    file_name = 'D:/DSlab/113_1/Ai/Federated-Learning-PyTorch-master/save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_{:.2f}_{:.2f}_ACC_INVERSE.pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs,train_accuracy_number,test_acc_number)
    with open(f'{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}]_{train_accuracy_number:.2f}_{test_acc_number:.2f}_ACC_INVERSE_acc.txt', 'w') as file:
        # Loop through the list and write each item on a new line
        for item in global_epoch_acc:
            file.write(str(item) + '\n')
    with open(f'{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}]_{train_accuracy_number:.2f}_{test_acc_number:.2f}_ACC_INVERSE_loss.txt', 'w') as file:
        # Loop through the list and write each item on a new line
        for item in global_epoch_loss:
            file.write(str(item) + '\n')  # Adding '\n' for a new line after each item
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    import datetime
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('D:/DSlab/113_1/Ai/Federated-Learning-PyTorch-master/save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_{:.2f}_{:.2f}_loss_ACC_INVERSE.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs,train_accuracy_number,test_acc_number))
    
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('D:/DSlab/113_1/Ai/Federated-Learning-PyTorch-master/save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_{:.2f}_{:.2f}_acc_ACC_INVERSE.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs,train_accuracy_number,test_acc_number))
