import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
import wandb
from datetime import datetime
import clip

from grasp_classifiers import *
import grasp_cls_pipeline_configs as configs
import data_processing as dp
import grasp_cls_utils as gcls_utils
import data_loading as dl
import nlp_processing as nlpp

def qualitative_data_analysis(dp_idx, gt_label, pred_label, pred_pos_probability, inputs, show_misclassified=False, lift_distance = None, show_correctly_classified=False, show_tp=True):
    if dp_idx <= 10:
        print("predicted positive probability:" + str(pred_pos_probability))
        print("pred label:" + str(pred_label))
        print("gt label:" + str(gt_label))
        if lift_distance is not None:
            print("lift distance:" + str(lift_distance))
        if inputs.size()[1] == 6:
            dp.render_rgb(inputs)
        else:
            dp.render_depth(inputs)

    if show_misclassified and pred_label != gt_label:
        print("### misclassification ###")
        print("predicted positive probability:" + str(pred_pos_probability))
        print("pred label:" + str(pred_label))
        print("gt label:" + str(gt_label))
        if lift_distance is not None:
            print("lift distance:" + str(lift_distance))
        if inputs.size()[1] == 6:
            dp.render_rgb(inputs)
        else:
            dp.render_depth(inputs)

    if show_correctly_classified and pred_label == gt_label:
        print("### correct classification ###")
        print("predicted positive probability:" + str(pred_pos_probability))
        print("pred label:" + str(pred_label))
        print("gt label:" + str(gt_label))
        if lift_distance is not None:
            print("lift distance:" + str(lift_distance))
        if inputs.size()[1] == 6:
            dp.render_rgb(inputs)
        else:
            dp.render_depth(inputs)

    if show_tp and pred_label == 1 and gt_label == 1:
        print("### true positive ###")
        print("predicted positive probability:" + str(pred_pos_probability))
        print("pred label:" + str(pred_label))
        print("gt label:" + str(gt_label))
        if lift_distance is not None:
            print("lift distance:" + str(lift_distance))
        if inputs.size()[1] == 6:
            dp.render_rgb(inputs)
        else:
            dp.render_depth(inputs)

def collect_confusion_matrix_datapoint(gt_label, pred_label, confusion_matrix_dict):
    if gt_label == 1:
        if pred_label == 1:
            confusion_matrix_dict["true_positives"] += 1
        elif pred_label == 0:
            confusion_matrix_dict["false_negatives"] += 1
    elif gt_label == 0:
        if pred_label == 1:
            confusion_matrix_dict["false_positives"] += 1
        if pred_label == 0:
            confusion_matrix_dict["true_negatives"] += 1

def generate_precision_recall_curve(gt_labels, pred_pos_probabilities, fold, val_iter, results_path):
    display = PrecisionRecallDisplay.from_predictions(y_true=gt_labels, y_pred=pred_pos_probabilities)
    plt.savefig(results_path+"precision_recall_curve_fold_"+str(fold)+"_num_"+str(val_iter))
    if configs.get_verbose():
        plt.show(block=True)

def calculate_precision_and_recall(confusion_matrix_dict):
    if confusion_matrix_dict["true_positives"] + confusion_matrix_dict["false_positives"] > 0:
        precision = confusion_matrix_dict["true_positives"]/(confusion_matrix_dict["true_positives"]+confusion_matrix_dict["false_positives"])
    else:
        precision = 0

    if confusion_matrix_dict["true_positives"] + confusion_matrix_dict["false_negatives"]> 0:
        recall = confusion_matrix_dict["true_positives"]/(confusion_matrix_dict["true_positives"]+confusion_matrix_dict["false_negatives"])
    else:
        recall = 0

    return precision, recall

def validate_classifier(model, val_files, result_dict, criterion=None, fold=-1, save_figure=True, results_path="", nlp_prompts=None, clip_model = None, object_type_dict=None, load_lift_distances=False):
    model.eval()
    tmp_val_dict = {}
    for val_iter, val_file_name in enumerate(val_files):
        ## LOAD DATASET ##
        if load_lift_distances:
            val_data, val_labels, lift_distances = dl.load_depth_dataset(dataset_path=val_file_name, load_lift_dist=load_lift_distances)
        else:
            val_data, val_labels = dl.load_depth_dataset(dataset_path=val_file_name, load_lift_dist=load_lift_distances)
        # remove unnessecary dimensions
        val_data = dp.prune_dimensions(val_data)
        val_labels = dp.prune_dimensions(val_labels)
        # initialize data loader
        val_loader = dl.haptic_dataset_to_data_loader(dataset=(val_data, val_labels))

        ## PREPARE FOR ITERATION THROUGH VAL LOADER ##
        pred_labels = []
        gt_labels = []
        pred_pos_probabilities = []
        confusion_matrix_dict = {
            "true_positives": 0,
            "false_negatives": 0,
            "false_positives": 0,
            "true_negatives": 0
        }
        object_type = val_file_name.split("/")[-1].split("_")[1]

        running_loss = 0
        average_loss = []
        ## ITERATE THROUGH VAL LOADER ##
        for idx, data in enumerate(val_loader, 0):
            #load datapoint
            inputs, label = data
            inputs = inputs.to(configs.get_device())
            gt_labels.append(label.detach().numpy()[0])

            ## GENERATE PREDICTION ##
            if configs.get_use_language_prompts():
                if configs.get_provide_raw_nlp_prompt():
                    nlp_prompt = nlpp.get_language_prompt(object_type, nlp_prompts)
                    prediction = model(inputs, nlp_prompt)
                else:
                    nlp_embedding = nlpp.get_language_embeddings(object_type, object_type_dict, nlp_prompts, clip_model)
                    prediction = model(inputs, nlp_embedding)
            else:
                prediction = model(inputs)

            # calculate validation loss
            if criterion is not None:
                loss = criterion(prediction, label.to(configs.get_device()))
                current_loss = loss.item()
                running_loss += current_loss
                if idx % 100 == 99: 
                    average_loss.append(running_loss/100)
                    wandb.log({"average validation loss in the last 100 steps": running_loss/100})
                    running_loss = 0
            

            # prediction to usable numpy
            prediction = dp.prune_dimensions(prediction.cpu().detach().numpy())
            # remove log so model acts as softmax (assumes NLL loss is usually used)
            prediction = np.exp(prediction)

            # extract predicted probability for label 1
            pred_pos_probability = prediction[1]
            pred_pos_probabilities.append(pred_pos_probability)

            # extract predicted label
            predicted_label = np.argmax(prediction)
            pred_labels.append(predicted_label)

            # potential qualitative prints for manual analysis
            if configs.get_verbose():
                if load_lift_distances:
                    lift_distance = lift_distances[idx]
                    qualitative_data_analysis(idx, label.item(), predicted_label, pred_pos_probability, inputs, lift_distance=lift_distance)
                else:
                    qualitative_data_analysis(idx, label.item(), predicted_label, pred_pos_probability, inputs)

            # collect confusion matrix data
            collect_confusion_matrix_datapoint(pred_label=predicted_label, gt_label=label.item(), confusion_matrix_dict=confusion_matrix_dict)

        # potentially save precision-recall curve
        if save_figure:
            generate_precision_recall_curve(gt_labels, pred_pos_probabilities, fold, val_iter, results_path)

        # calculate precision and recall
        precision, recall = calculate_precision_and_recall(confusion_matrix_dict)

        gcls_utils.dict_list_append("precision", precision, tmp_val_dict)
        gcls_utils.dict_list_append("recall", recall, tmp_val_dict)
        gcls_utils.dict_list_append("tp", confusion_matrix_dict["true_positives"], tmp_val_dict)
        gcls_utils.dict_list_append("tn", confusion_matrix_dict["true_negatives"], tmp_val_dict)
        gcls_utils.dict_list_append("fp", confusion_matrix_dict["false_positives"], tmp_val_dict)
        gcls_utils.dict_list_append("fn", confusion_matrix_dict["false_negatives"], tmp_val_dict)

    gcls_utils.stack_validation_averages(result_dict, tmp_val_dict) # collapse temporary entries and add them to global results
    model.train()

def reduce_train_list_dim_for_wandb(input_list):
    result_list = []
    if isinstance(input_list[0],list) or (isinstance(input_list, np.ndarray) and len(np.shape(input_list)) > 1):
        for sublist in input_list:
            if isinstance(sublist,list):
                result_list = result_list + sublist
            else:
                result_list = result_list + sublist.tolist()
        return result_list
    else:
        return input_list

def wandb_setup(optimizer, dnt_start, depth_grasp_classifier, train_files, val_files, fold, scheduler, results_path, nlp_classifier_name=""):
    group_name = results_path.split("/")[-2]
    config={
        "optimizer type": type(optimizer),
        "loss type": type(configs.get_loss_criterion()),
        "learning_rate": configs.get_learning_rate(),
        "model": depth_grasp_classifier.name,
        "training datasets": reduce_train_list_dim_for_wandb(train_files),
        "validation datasets": val_files,
        "epochs": configs.get_num_epochs(),
        "crossval fold": fold,
        "total folds": configs.get_num_folds(),
        "pixel reduction factor": configs.get_pixel_reduction_factor(),
        "start time and date": dnt_start,
        "scheduler type": type(scheduler)
    }

    if configs.get_use_language_prompts():
        config["clip model"] = configs.get_clip_model_name()
        config["language prompts"] = configs.get_language_prompts_path()

    if isinstance(scheduler, optim.lr_scheduler.ExponentialLR):
        config["scheduler gamma"] = configs.get_gamma()

    if isinstance(configs.get_loss_criterion() , nn.NLLLoss):
        config["NLL weights"] = configs.get_criterion_args()["weight"]

    if nlp_classifier_name is not None:
        config["nlp classifier"] = nlp_classifier_name

    if configs.get_load_cnn():
        config["cnn path"] = configs.get_cnn_path()

    if configs.get_dataset_name() is not None:
        config["dataset name"] = configs.get_dataset_name()

    run_name = "fold_" + str(fold)

    wandb.init(
        # set the wandb project where this run will be logged
        project="haptic-exploration-with-nlp",
        # track hyperparameters and run metadata
        config=config,
        name=run_name,
        group=group_name
    )     

def log_training_step_metrics(epoch, current_loss, labels, loss_vals_gtf, loss_vals_gtt, average_loss, i, running_loss):
    # print statistics
    #if i % 10 == 9:    # print every 2000 mini-batches
    if configs.get_verbose():
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {current_loss:.3f}')
    if configs.get_batch_size() == 1:
        if labels[0].item() == 0:
            loss_vals_gtf.append(current_loss)
            wandb.log({"ground truth negative loss": current_loss})
        else:
            loss_vals_gtt.append(current_loss)
            wandb.log({"ground truth positive loss": current_loss})

    if i % 100 == 99: 
        average_loss.append(running_loss/100)
        wandb.log({"average overall loss in the last 100 steps": running_loss/100})
        running_loss = 0

    return running_loss       

def train_test_depth_pipeline(dataset_path="", dnt_start="none", results_path=""):
    criterion = configs.get_loss_criterion()
    clip_nlp_model = None
    if configs.get_use_language_prompts():
        clip_nlp_model, _ = clip.load(configs.get_clip_model_name(), device=configs.get_device())
        nlp_promts = np.load(configs.get_language_prompts_path())

    # TODO: Function to set this to pre established train/val splits for more comparability  
    dataset_paths = dl.get_dataset_paths(dataset_path)
    object_type_dict = nlpp.generate_object_type_dict(dataset_paths)

    random.shuffle(dataset_paths)

    if configs.get_provide_raw_nlp_prompt():
        train_val_prompts = nlpp.split_nlp_prompt_dict(configs.get_num_folds())

    validation_results = {}
    for fold in range(configs.get_num_folds()):
        loss_vals_gtf = []
        loss_vals_gtt = []
        average_loss = []
        snapshot_count = 0
        current_best_f1 = -1 # always save at least first model

        nlp_classifier_name = None
        if configs.get_load_pretrained_model():
            depth_grasp_classifier = torch.load(configs.get_model_path())
        else:
            if configs.get_provide_raw_nlp_prompt():
                nlp_classifier = torch.load(configs.get_nlp_classifier_path())
                nlp_classifier_name = nlp_classifier.name
                if configs.get_use_raw_clip_model():
                    depth_grasp_classifier = Grasp_Classifier_Raw_CLIP()
                else:
                    depth_grasp_classifier = Depth_Grasp_Classifier_v3_nrm_ltag_col(nlp_classifier)
            elif configs.get_use_language_prompts() and not configs.get_calculate_language_embedding():
                depth_grasp_classifier = Depth_Grasp_Classifier_v3_nrm_ltagp_col()
            elif configs.get_load_cnn():
                cnn = configs.get_cnn()
                depth_grasp_classifier = Depth_Grasp_Classifier_v3_norm_col_preset_CNN(cnn)
            else:
                depth_grasp_classifier = Depth_Grasp_Classifier_v3_norm_col2()

        if configs.get_frozen_cnn():
            for param in depth_grasp_classifier.cnn_feature_extract:
                param.requires_grad = False

        if configs.get_frozen_clip() and isinstance(depth_grasp_classifier, Depth_Grasp_Classifier_v3_nrm_ltag_col):
            for param in depth_grasp_classifier.nlp_model.clip.transformer.resblocks:
                param.requires_grad = False
        elif configs.get_frozen_clip() and isinstance(depth_grasp_classifier, Grasp_Classifier_Raw_CLIP):
            for param in depth_grasp_classifier.nlp_model.modules():
                param.requires_grad = False

        depth_grasp_classifier.to(configs.get_device())
        optimizer = optim.SGD(depth_grasp_classifier.parameters(), lr=configs.get_learning_rate())
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=configs.get_gamma())
        if configs.get_no_crossval():
            #train on entire dataset
            tmp1, tmp2 = dl.split_file_list_evenly_groupings(dataset_paths, configs.get_num_folds(), fold)
            train_files = tmp2.tolist()
            append_chunk = tmp1[0:int(len(tmp1)/3)]
            if len(append_chunk) > 0:
                train_files.append(tmp1[0:int(len(tmp1)/3)])
            append_chunk = tmp1[int(len(tmp1)/3):2*int(len(tmp1)/3)]
            if len(append_chunk) > 0:
                train_files.append(append_chunk)
            append_chunk = tmp1[2*int(len(tmp1)/3):-1]
            if len(append_chunk) > 0:
                train_files.append(append_chunk)
            val_files = None
        else:
            if configs.get_predefined_train_val_path() is not None:
                val_files, train_files = dl.load_predefined_train_val(fold)
            else:
                val_files, train_files = dl.split_file_list_evenly_groupings(dataset_paths, configs.get_num_folds(), fold)

        wandb_setup(optimizer, dnt_start, depth_grasp_classifier, train_files, val_files, fold, scheduler, results_path, nlp_classifier_name)

        print(f"###### fold: {fold} ######")

        depth_grasp_classifier.train()

        for epoch in range(configs.get_num_epochs()):
            for file_idx, multi_file_train_chunk in enumerate(train_files):
                print(f"--- {multi_file_train_chunk} ---")

                train_loader = dl.load_multi_file_dataset(multi_file_train_chunk)

                running_loss = 0
                
                for i, batch in enumerate(train_loader, 0):
                    inputs, labels, object_types = batch
                    if np.shape(inputs)[0] == 1: # 1 dim batches throw batchnorm for a loop. Simply skip these batches
                        continue
                    inputs, labels = inputs.to(configs.get_device()), labels.to(configs.get_device())

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    if configs.get_use_language_prompts():
                        if isinstance(depth_grasp_classifier, Depth_Grasp_Classifier_v3_nrm_ltag_col) or isinstance(depth_grasp_classifier, Grasp_Classifier_Raw_CLIP):
                            language_prompt = nlpp.get_language_prompt(object_types, train_val_prompts[fold]["train"])
                            outputs = depth_grasp_classifier(inputs, language_prompt) 
                        else:
                            language_embedding = nlpp.get_language_embeddings(object_types, object_type_dict, nlp_promts, clip_nlp_model)
                            outputs = depth_grasp_classifier(inputs, language_embedding)
                    else:
                        outputs = depth_grasp_classifier(inputs)
                        
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    current_loss = loss.item()
                    running_loss += current_loss

                    running_loss = log_training_step_metrics(epoch, current_loss, labels, loss_vals_gtf, loss_vals_gtt, average_loss, i, running_loss)

                if configs.get_regular_save():
                    if file_idx + epoch*len(train_files) % 5 == 0: # every 5 files: make a snapshot that is permanent
                        torch.save(depth_grasp_classifier, results_path+depth_grasp_classifier.name+"_model_snapshot_"+str(snapshot_count)+"_fold_"+str(fold))
                        snapshot_count += 1
                    else: # otherwise make a snapshot that is temporary in case of crash
                        torch.save(depth_grasp_classifier, results_path+depth_grasp_classifier.name+"_model_snapshot_"+str(snapshot_count)+"_fold_"+str(fold))
            
                # run validation after every "file batch"
                if not configs.get_no_crossval():
                    intermediate_validation_results = {}
                    if configs.get_use_language_prompts():
                        if configs.get_provide_raw_nlp_prompt():
                            raw_nlp_prompts = train_val_prompts[fold]["train"]
                            validate_classifier(depth_grasp_classifier, val_files, intermediate_validation_results, criterion=criterion, fold=fold, save_figure=False, results_path=results_path, nlp_prompts=raw_nlp_prompts, clip_model=clip_nlp_model, object_type_dict=object_type_dict)
                        else:
                            validate_classifier(depth_grasp_classifier, val_files, intermediate_validation_results, criterion=criterion, fold=fold, save_figure=False, results_path=results_path, nlp_prompts=nlp_promts, clip_model=clip_nlp_model, object_type_dict=object_type_dict)
                    else:
                        validate_classifier(depth_grasp_classifier, val_files, intermediate_validation_results, criterion=criterion, fold=fold, save_figure=False, results_path=results_path)
                    wandb.log(gcls_utils.info_dict_to_wandb_format(intermediate_validation_results))

                    precision = intermediate_validation_results["precision"]
                    recall = intermediate_validation_results["recall"]
                    if precision == 0 and recall == 0:
                        f1_score = 0
                    else:
                        f1_score = 2*(precision*recall)/(precision+recall)

                    if f1_score > current_best_f1:
                        torch.save(depth_grasp_classifier, results_path+depth_grasp_classifier.name+"_best_model_fold_"+str(fold))
                        current_best_f1 = f1_score

                    wandb.log({"f1 score": f1_score})

                    wandb.log({"current learning rate": scheduler.get_last_lr()[-1]})
            if configs.get_epoch_save():
                torch.save(depth_grasp_classifier, results_path+depth_grasp_classifier.name+"_model_snapshot_epoch_"+str(epoch)+"_fold_"+str(fold))
            scheduler.step()

        # Redundant, covered by wandb
        # fig, ax = plt.subplots(3)
        # ax[0].plot(loss_vals_gtf)
        # ax[1].plot(loss_vals_gtt)
        # ax[2].plot(average_loss)
        # plt.savefig(PATH[CURRENT_DEVICE]+"training_plot"+str(plot_count)+"_fold_"+str(fold))
        # plot_count += 1
        # if VERBOSE:
        #    plt.show()

        if configs.get_no_crossval():
            break

        best_model = torch.load(results_path+depth_grasp_classifier.name+"_best_model_fold_"+str(fold))
        tmp_validation_results = {}

        if configs.get_use_language_prompts():
            validate_classifier(best_model, val_files, tmp_validation_results, criterion=criterion, fold=fold, results_path=results_path, nlp_prompts=nlp_promts, clip_model=clip_nlp_model, object_type_dict=object_type_dict)
        else:
            validate_classifier(best_model, val_files, tmp_validation_results, criterion=criterion, fold=fold, results_path=results_path)
        wandb.log(gcls_utils.info_dict_to_wandb_format(intermediate_validation_results)) # last logged model in each fold is the best performing model
        for key in tmp_validation_results.keys():
            gcls_utils.dict_list_append(key, tmp_validation_results[key], validation_results)

        wandb.finish()

    wandb.finish()
    #print("Precision Values: " + str(validation_results["precision"]))
    #print("Recall Values: " + str(validation_results["recall"]))
    #print("average Precision:" + str(statistics.mean(validation_results["precision"])))
    #print("average Recall:" + str(statistics.mean(validation_results["recall"])))

def load_and_eval(model_path=configs.get_model_path(), val_file_paths=configs.get_val_file_paths(), results_path="", load_nlp_prompts=True):
    if isinstance(model_path, list):
        for path in model_path:
            load_and_eval(model_path=path, val_file_paths=val_file_paths, results_path=results_path, load_nlp_prompts=load_nlp_prompts)
    else:
        validation_results = {}
        model = torch.load(model_path)
        if load_nlp_prompts:
            # very hacky solution to loading the nlp prompts that were not used for training 
            train_val_prompts = nlpp.split_nlp_prompt_dict(configs.get_num_folds())
            nlp_prompts = train_val_prompts[0]["val"]
            validate_classifier(model, val_file_paths, validation_results, results_path=results_path, load_lift_distances=False, nlp_prompts=nlp_prompts)
        else:
            validate_classifier(model, val_file_paths, validation_results, results_path=results_path, load_lift_distances=False)

        print(validation_results)

        precision = validation_results["precision"][0]
        recall = validation_results["recall"][0]

        if precision+recall != 0:
            f1 = 2*((precision*recall)/(precision+recall))
        else:
            f1 = 0.0

        print(f"-> f1: {f1}")

def main():
    if not configs.get_train():
        os.environ['WANDB_MODE'] = 'offline' # uncomment when making development runs
    random.seed(42)
    np.random.seed(seed=42)
    now = datetime.now()
    current_time_and_date = now.strftime("%m.%d.%y_%H:%M:%S")
    results_folder_name = "results_run_" + current_time_and_date
    path = os.path.join(configs.get_path(), results_folder_name)
    results_folder_name = results_folder_name + "/"
    os.mkdir(path)

    if configs.get_train():
        train_test_depth_pipeline(dataset_path=configs.get_dataset_path(), dnt_start=current_time_and_date, results_path=configs.get_path()+results_folder_name)
    else:
        load_and_eval(results_path=configs.get_path()+results_folder_name)

if __name__ == '__main__':
    main()






