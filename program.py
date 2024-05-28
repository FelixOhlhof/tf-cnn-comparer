import argparse
parser = argparse.ArgumentParser(description='Training and comparison of different CNNs')

import os
import util
from datetime import datetime
from Classifier import Classifier

if __name__ == "__main__":
    settings = util.load_general_settings()
    classifier = util.get_classifier()
    results_dir = util.create_new_results_folder()

    if(not settings["use_gpu"]):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # nessesary for the seed
    
    for c in classifier:
        c.train()  
        c.write_report(results_dir)
        c.save_validation_loss_plot(results_dir)
        c.save_model(results_dir)
    util.save_roc_curve(classifier, results_dir)