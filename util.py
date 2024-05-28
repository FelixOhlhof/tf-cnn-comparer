
import configparser
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