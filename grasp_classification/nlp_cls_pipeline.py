import torch
import numpy as np
import torch.optim as optim
import os
import random
import wandb
from datetime import datetime
import json

import nlp_cls_pipeline_configs as configs
import nlp_classifiers as nlp_cls
import data_loading as dl
import data_processing as dp
import grasp_cls_utils as gcls_utils

def wandb_setup(optimizer, dnt_start, nlp_classifier, fold, scheduler, results_path):
    group_name = results_path.split("/")[-2]
    config={
        "optimizer type": type(optimizer),
        "loss type": type(configs.get_loss_criterion()),
        "learning_rate": configs.get_learning_rate(),
        "model": nlp_classifier.name,
        "epochs": configs.get_num_epochs(),
        "crossval fold": fold,
        "total folds": configs.get_num_folds(),
        "start time and date": dnt_start,
        "scheduler type": type(scheduler),
        "clip model": configs.get_clip_model_name(),
        "dataset": configs.get_nlp_dataset_path(),
        "random_seed": configs.get_random_seed()
    }

    if isinstance(scheduler, optim.lr_scheduler.ExponentialLR):
        config["scheduler gamma"] = configs.get_gamma()

    run_name = "fold_" + str(fold)

    wandb.init(
        # set the wandb project where this run will be logged
        project="nlp_object_classifier",
        # track hyperparameters and run metadata
        config=config,
        name=run_name,
        group=group_name
    )   

def validate_classifier(model, val_data, val_labels, result_dict):
    model.eval()
    correct_classifications = 0

    # initialize data loader
    val_loader = dl.nlp_dataset_to_data_loader(dataset=(val_data, val_labels))

    ## PREPARE FOR ITERATION THROUGH VAL LOADER ##
    f = open(configs.get_object_type_json_path(),mode="r")
    label_dict = json.load(f)

    correct_per_class_dict = {}
    for label in val_labels:
        if not label in correct_per_class_dict.keys():
            correct_per_class_dict[label] = 0

    ## ITERATE THROUGH VAL LOADER ##
    for data in val_loader:
        #load datapoint
        inputs, label = data
        inputs = inputs[0]

        ## GENERATE PREDICTION ##
        prediction = model(inputs)

        # prediction to usable numpy
        prediction = dp.prune_dimensions(prediction.cpu().detach().numpy())

        # extract predicted label
        predicted_label = np.argmax(prediction)

        if predicted_label == np.argmax(label.detach().numpy()[0]):
            class_label = [key for key in label_dict.keys() if predicted_label == label_dict[key]][0]
            correct_per_class_dict[class_label] += 1
            correct_classifications += 1

    accuracy = correct_classifications/len(val_labels)
    gcls_utils.dict_list_append("accuracy", accuracy, result_dict)
    gcls_utils.dict_list_append("correct classifications", correct_classifications, result_dict)
    for key in correct_per_class_dict.keys():
        gcls_utils.dict_list_append(f"correct classification for {key}", correct_per_class_dict[key], result_dict)
    
    model.train()

def train_test_depth_pipeline(dnt_start="none", results_path=""):
    criterion = configs.get_loss_criterion()
    dataset = np.load(configs.get_nlp_dataset_path())
    labels = dataset["object type"]
    dataset = dataset["nlp prompt"]
    crosval_split_dict = dl.nlp_train_test_split(dataset, labels, configs.get_num_folds())

    validation_results = {}
    for fold in range(configs.get_num_folds()):
        current_best_accuracy = -1 # always save at least first model
        nlp_classifier = nlp_cls.Nlp_classifier2()
        optimizer = optim.SGD(nlp_classifier.parameters(), lr=configs.get_learning_rate())
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=configs.get_gamma())

        train_data = crosval_split_dict[f"train_{fold}_data"]
        train_labels = crosval_split_dict[f"train_{fold}_labels"]
        val_data = crosval_split_dict[f"val_{fold}_data"]
        val_labels = crosval_split_dict[f"val_{fold}_labels"]

        wandb_setup(optimizer, dnt_start, nlp_classifier, fold, scheduler, results_path)

        print(f"###### fold: {fold} ######")

        nlp_classifier.train()
        nlp_classifier.to(configs.get_device())

        for epoch in range(configs.get_num_epochs()):

            train_loader = dl.nlp_dataset_to_data_loader(dataset=(train_data, train_labels))
            running_loss = 0
            
            for i, batch in enumerate(train_loader, 0):
                inputs, labels = batch
                inputs, labels = inputs[0], labels.to(configs.get_device())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = nlp_classifier(inputs)
                    
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

                current_loss = loss.item()
                running_loss += current_loss

                wandb.log({"current loss": current_loss})

            # run validation after every "file batch"
            intermediate_validation_results = {}
            validate_classifier(nlp_classifier, val_data, val_labels, intermediate_validation_results)
            
            wandb.log(gcls_utils.info_dict_to_wandb_format(intermediate_validation_results))

            accuracy = intermediate_validation_results["accuracy"]

            if accuracy > current_best_accuracy:
                torch.save(nlp_classifier, results_path+nlp_classifier.name+"_best_model_fold_"+str(fold))
                current_best_accuracy = accuracy

            wandb.log(gcls_utils.info_dict_to_wandb_format(intermediate_validation_results))

            wandb.log({"current learning rate": scheduler.get_last_lr()[-1]})

            scheduler.step()

        best_model = torch.load(results_path+nlp_classifier.name+"_best_model_fold_"+str(fold))
        tmp_validation_results = {}

        validate_classifier(best_model, val_data, val_labels, tmp_validation_results)

        wandb.log(gcls_utils.info_dict_to_wandb_format(tmp_validation_results)) # last logged model in each fold is the best performing model
        for key in tmp_validation_results.keys():
            gcls_utils.dict_list_append(key, tmp_validation_results[key], validation_results)

        wandb.finish()

        if configs.get_no_crossval():
            break

def main():
    #os.environ['WANDB_MODE'] = 'offline' # uncomment when making development runs
    if configs.get_random_seed() is not None:
        random.seed(configs.get_random_seed())
        np.random.seed(seed=configs.get_random_seed())

    now = datetime.now()
    current_time_and_date = now.strftime("%m.%d.%y_%H:%M:%S")
    results_folder_name = "nlp_cls_results_run_" + current_time_and_date
    path = os.path.join(configs.get_path(), results_folder_name)
    results_folder_name = results_folder_name + "/"
    os.mkdir(path)

    train_test_depth_pipeline(dnt_start=current_time_and_date, results_path=configs.get_path()+results_folder_name)

if __name__ == '__main__':
    main()