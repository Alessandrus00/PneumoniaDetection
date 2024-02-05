import matplotlib.pyplot as plt
import numpy as np
import os
import random
from torch.utils.data import WeightedRandomSampler
import pandas as pd

def retrieve_metrics(checkpoint_path):
    directory_path = os.path.join(checkpoint_path,"logs")
    entries = os.listdir(directory_path)
    # Filter out only the subdirectories
    versions = [entry for entry in entries if os.path.isdir(os.path.join(directory_path, entry))]
    df_list = []
    for version in versions:
        metrics_path = os.path.join(directory_path, version, "metrics.csv")
        if os.path.isfile(metrics_path):
            df_list.append(pd.read_csv(metrics_path))
    
    df = pd.concat(df_list)

    return df['train_loss_epoch'].dropna(), df['train_acc_epoch'].dropna(), df['val_loss_epoch'].dropna(), df['val_acc_epoch'].dropna()


def plot_results(checkpoint_path):
    train_losses, train_accs, val_losses, val_accs = retrieve_metrics(checkpoint_path)

    n_epochs = len(train_losses)
    
    # Plot results after trainind ends
    plt.rc('xtick',labelsize=13)
    plt.rc('ytick',labelsize=13)

    plt.figure(figsize=(20, 6))
    _ = plt.subplot(1,2,1)
    plt.plot(np.arange(n_epochs) + 1, train_losses, 'o-', linewidth=3)
    plt.plot(np.arange(n_epochs) + 1, val_losses, 'o-', linewidth=3)
    _ = plt.legend(['Train', 'Validation'], fontsize=12.5)
    plt.grid('on'), plt.xlabel('Epoch',fontsize=17), plt.ylabel('Loss',fontsize=17)
    plt.title('Training and validation loss',fontsize="19")

    _ = plt.subplot(1,2,2)
    plt.plot(np.arange(n_epochs) + 1, train_accs, 'o-', linewidth=3)
    plt.plot(np.arange(n_epochs) + 1, val_accs, 'o-', linewidth=3)
    _ = plt.legend(['Train', 'Validation'], fontsize=12.5)
    plt.grid('on'), plt.xlabel('Epoch', fontsize=17), plt.ylabel('Accuracy',fontsize=17)
    plt.title('Training and validation accuracy',fontsize="19")
    #plt.savefig(os.path.join(trainer.logger.log_dir,'learning_curves.pdf'))
    plt.show()


def extract_patient_ids(filename):
    patient_id = filename.split('_')[0].replace("person", "")
    return patient_id


def split_file_names_meta(input_folder,n_samples=200):
    random.seed(27)
    pneumonia_patient_ids = set([extract_patient_ids(fn) for fn in os.listdir(os.path.join(input_folder, 'PNEUMONIA'))])
    pneumonia_meta_patient_ids = random.sample(pneumonia_patient_ids, int(45))
    
    print(len(pneumonia_meta_patient_ids))

    pneumonia_train_filenames = []
    pneumonia_meta_filenames = []

    for filename in os.listdir(os.path.join(input_folder, 'PNEUMONIA')):
        patient_id = extract_patient_ids(filename)
        if patient_id in pneumonia_meta_patient_ids:
            pneumonia_meta_filenames.append(os.path.join(input_folder, 'PNEUMONIA', filename))
        else:
            pneumonia_train_filenames.append(os.path.join(input_folder, 'PNEUMONIA', filename))
    
    print(len(pneumonia_meta_filenames))
    
    # Normal (by file, no patient information in file names)
    normal_filenames  = [os.path.join(input_folder, 'NORMAL', fn) for fn in os.listdir(os.path.join(input_folder, 'NORMAL'))]
    random.seed(27)
    normal_meta_filenames = random.sample(normal_filenames, int(100))
    normal_train_filenames = list(set(normal_filenames)-set(normal_meta_filenames))

    #print(pneumonia_meta_patient_ids)


    meta_filenames = pneumonia_meta_filenames + normal_meta_filenames

    return pneumonia_train_filenames, normal_train_filenames, meta_filenames


def split_file_names2(pneumonia_train_filenames, normal_train_filenames, val_split_perc):
    pneumonia_val_filenames = random.sample(pneumonia_train_filenames, int(val_split_perc*len(pneumonia_train_filenames)))
    pneumonia_train_filenames = list(set(pneumonia_train_filenames) - set(pneumonia_val_filenames))

    normal_val_filenames = random.sample(normal_train_filenames, int(val_split_perc*len(normal_train_filenames)))
    normal_train_filenames = list(set(normal_train_filenames) - set(normal_val_filenames))

    return (pneumonia_train_filenames + normal_train_filenames), (pneumonia_val_filenames + normal_val_filenames)




def split_file_names(input_folder, val_split_perc):
    # Pneumonia files contain patient id, so we group split them by patient to avoid data leakage
    pneumonia_patient_ids = set([extract_patient_ids(fn) for fn in os.listdir(os.path.join(input_folder, 'PNEUMONIA'))])
    pneumonia_val_patient_ids = random.sample(list(pneumonia_patient_ids), int(val_split_perc * len(pneumonia_patient_ids)))

    pneumonia_val_filenames = []
    pneumonia_train_filenames = []

    for filename in os.listdir(os.path.join(input_folder, 'PNEUMONIA')):
        patient_id = extract_patient_ids(filename)
        if patient_id in pneumonia_val_patient_ids:
            pneumonia_val_filenames.append(os.path.join(input_folder, 'PNEUMONIA', filename))
        else:
            pneumonia_train_filenames.append(os.path.join(input_folder, 'PNEUMONIA', filename))

    # Normal (by file, no patient information in file names)
    normal_filenames  = [os.path.join(input_folder, 'NORMAL', fn) for fn in os.listdir(os.path.join(input_folder, 'NORMAL'))]
    normal_val_filenames = random.sample(normal_filenames, int(val_split_perc * len(normal_filenames)))
    normal_train_filenames = list(set(normal_filenames)-set(normal_val_filenames))

    train_filenames = pneumonia_train_filenames + normal_train_filenames
    val_filenames = pneumonia_val_filenames + normal_val_filenames

    return train_filenames, val_filenames
    

def create_weighted_sampler(dataset):
    targets = dataset.targets
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    weights = [class_weights[label] for label in targets]
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler