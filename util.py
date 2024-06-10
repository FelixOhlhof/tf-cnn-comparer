
import os
import tensorflow as tf
import random
import configparser
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from typing import List
from sklearn.metrics import roc_curve, auc
from pathlib import Path
from datetime import datetime
from Classifier import Classifier

config = configparser.ConfigParser()
config.read("config.ini")


def load_general_settings():
    for section_name in config.sections():
        if section_name == "General":
            use_gpu = config.getboolean("General","USE_GPU")
            use_seed = config.getboolean("General","USE_SEED")
            log_file = config["General"]["LOG_FILE"]
            return dict(use_gpu=use_gpu, log_file=log_file, use_seed=use_seed)

def get_classifier(general_settings) -> List[Classifier]:
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
                epochs=epochs,
                c_label=section_name,
                general_settings=general_settings)
            )
    return classifier

def create_new_results_folder():
    # results_folder = os.path.join(Path.cwd(), "results")
    results_folder = os.path.join("results", datetime.now().strftime("%d.%m.%Y %H.%M.%S"))
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    return results_folder

def load_image(img_path, show=False):
    img = tf.keras.utils.load_img(
        img_path,
        color_mode="rgb",
        target_size=(200, 200),
        interpolation="nearest",
        keep_aspect_ratio=False,
    )
    img_tensor = tf.keras.utils.img_to_array(img)     
    img_tensor = np.array([img_tensor])  
    img_tensor /= 255.           

    return img_tensor

def save_roc_curve(classifier, path):
    colors = ["red", "green", "blue", "yellow", "olive", "pink"]
    plt.figure()	

    for c in classifier:
        fpr, tpr, _ = roc_curve(c.y_true, c.y_pred)
        roc_auc = auc(fpr, tpr)

        lw = 2
        line_color = random.choice(colors)
        colors.remove(line_color)
        plt.plot(fpr, tpr, color=line_color, lw=lw, label=f"{c.c_label}_{c.model_name} AUC {roc_auc}")
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    graph = os.path.join(path, f"{c.c_label}_{c.model_name}_roc.png")
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(graph, dpi=600)
    plt.close()