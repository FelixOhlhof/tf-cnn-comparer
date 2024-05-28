
import configparser
import matplotlib.colors as mcolors
import numpy as np
import random
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
from typing import List

from pathlib import Path
from time import sleep
from datetime import datetime
from Classifier import Classifier

config = configparser.ConfigParser()
config.read("config.ini")


def load_general_settings():
    for section_name in config.sections():
        if section_name == "General":
            use_gpu = config.getboolean("General","USE_GPU")
            return dict(use_gpu=use_gpu)

def get_classifier() -> List[Classifier]:
    classifier = []

    for section_name in config.sections():
        if section_name == "General":
            continue
        for line in config[section_name]:
            model_name = config[section_name]["MODEL_NAME"]
            train_dir = config[section_name]["TRAIN_DATA_DIR"]
            test_dir = config[section_name]["TEST_FATA_DIR"]
            validation_dir = config[section_name]["VALIDATION_DATA_DIR"]
            epochs=(int)(config[section_name]["EPOCHS"])  
            batch_size = (int)(config[section_name]["BATCH_SIZE"])
            seed = (int)(config[section_name]["SEED"])
            img_height = (int)(config[section_name]["IMG_HEIGHT"])
            img_width = (int)(config[section_name]["IMG_WIDTH"])
        classifier.append(
            Classifier(
                train_dir=train_dir, 
                test_dir=test_dir, 
                val_dir=validation_dir, 
                seed=seed, 
                batch_size=batch_size, 
                model_name=model_name, 
                img_height=img_height, 
                img_width=img_width, 
                epochs=epochs)
            )
    return classifier

def create_new_results_folder():
    results_folder = os.path.join(Path.cwd(), "results")
    results_folder = os.path.join(results_folder, datetime.now().strftime("%d.%m.%Y %H.%M.%S"))
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    return results_folder

def save_roc_curve(classifier, path):
    colors = list(mcolors.CSS4_COLORS)
    plt.figure()	

    for c in classifier:
        preds = c.model.predict(c.test_ds, verbose=1)
        fpr, tpr, _ = roc_curve(np.concatenate([y for x, y in c.test_ds], axis=0), preds)
        roc_auc = auc(fpr, tpr)
        lw = 2
        line_color = random.choice(colors)
        colors.remove(line_color)
        plt.plot(fpr, tpr, color=line_color, lw=lw, label=f"{c.model_name} AUC {roc_auc}")
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    graph = os.path.join(path, f"{c.model_name}_roc.png")
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(graph, dpi=600)
    plt.close()