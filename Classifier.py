import tensorflow as tf
import numpy as np
import random as rn
import models
import matplotlib.pyplot as plt
import csv
import os

from tensorflow import keras
from contextlib import redirect_stdout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Classifier():
  def __init__(self, train_dir, test_dir, val_dir, seed, batch_size, model_name, img_height, img_width, epochs, c_label, general_settings):
    self.classes = ["morph", "no_morph"]
    self.train_dir = train_dir
    self.val_dir = val_dir
    self.test_dir = test_dir
    if(general_settings["use_seed"]):
      self.seed = seed
    else:
      self.seed = int(np.random.rand() * (2**32 - 1))
    np.random.seed(self.seed)
    rn.seed(self.seed)
    tf.random.set_seed(self.seed)
    self.batch_size = batch_size
    self.img_height = img_height
    self.img_width = img_width
    self.callbacks_list = None
    self.epochs = epochs
    self.train_ds, self.val_ds, self.test_ds = self.get_datasets()
    self.model_name = model_name
    self.c_label = c_label
    model_method = getattr(models, model_name)
    self.model = model_method(img_height, img_width)
    self.general_settings = general_settings
    self.c_description = f"Model: {self.model_name}  Batch size: {self.batch_size}  Seed: {self.seed}  Dropout Layer: {len(['y' for l in self.model.layers if l.name == 'dropout']) > 0}  Optimizer: {self.model.optimizer.name}  Learning Rate: {str(round(self.model.optimizer.learning_rate.numpy(), 4))}  Epochs: {self.epochs}"
    

  def get_datasets(self):
    train_ds = tf.keras.utils.image_dataset_from_directory(
      self.train_dir,
      seed=self.seed,
      labels='inferred',
      image_size=(self.img_height, self.img_width),
      batch_size=self.batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
      self.val_dir,
      seed=self.seed,
      labels='inferred',
      image_size=(self.img_height, self.img_width),
      batch_size=self.batch_size)
    test_ds = tf.keras.utils.image_dataset_from_directory(
      self.test_dir,
      seed=self.seed,
      labels='inferred',
      image_size=(self.img_height, self.img_width),
      batch_size=self.batch_size,
      shuffle=False)
    return train_ds, val_ds, test_ds

  def train(self):
    self.history = self.model.fit(
      self.train_ds,
      validation_data=self.val_ds,
      epochs=self.epochs,
      callbacks= self.callbacks_list
    )
    self.acc_name = list(self.history.history)[0]
    self.loss_name = list(self.history.history)[1]
    self.acc_val_name = list(self.history.history)[2]
    self.loss_val_name = list(self.history.history)[3]
    return self.history

  def evaluate(self):
    self.y_pred = self.model.predict(self.test_ds, verbose=1)
    y_true = [y for _,y in  self.test_ds]
    self.y_true = np.concatenate(y_true , axis = 0)

  def save_model_summary(self, path):
    with open(os.path.join(path, f"{self.c_label}_{self.model_name}_summary.txt"), 'w') as f:
      with redirect_stdout(f):
          self.model.summary()

  def save_confusion_matrix(self, path):
    y_pred = (self.y_pred > 0.5).astype(int)
    cm = confusion_matrix(self.y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.test_ds.class_names)
    disp.plot(cmap=plt.cm.Blues)

    plt.suptitle(self.c_description, fontsize=6) 
    plt.title(f'Confusion Matrix, Threshold: 0.5')
    plt.savefig(os.path.join(path, f"{self.c_label}_{self.model_name}_cm.png"), dpi=600)

  def save_validation_loss_plot(self, path):
      #visualize training results
      acc = self.history.history[self.acc_name]
      val_acc = self.history.history[self.acc_val_name]

      loss = self.history.history[self.loss_name]
      val_loss = self.history.history[self.loss_val_name]

      epochs_range = range(self.epochs)

      plt.figure(figsize=(8, 8))
      plt.suptitle(self.c_description, fontsize=10) 

      plt.subplot(1, 2, 1)
      plt.plot(epochs_range, acc, label='Training Accuracy')
      plt.plot(epochs_range, val_acc, label='Validation Accuracy')
      plt.legend(loc='lower right')
      plt.title('Training and Validation Accuracy')

      plt.subplot(1, 2, 2)
      plt.plot(epochs_range, loss, label='Training Loss')
      plt.plot(epochs_range, val_loss, label='Validation Loss')
      plt.legend(loc='upper right')
      plt.title('Training and Validation Loss')
      mng = plt.get_current_fig_manager()
      mng.full_screen_toggle()
      plt.savefig(os.path.join(path, f"{self.c_label}_{self.model_name}_val_loss.png"), dpi=600)
      plt.close()

  def save_model(self, path):
    self.model.save(os.path.join(path, f"{self.c_label}_{self.model_name}.keras"))
  
  def write_report(self, results_path):
    evaluation = self.model.evaluate(self.test_ds)
    base_columns = None

    if not os.path.isfile(self.general_settings["log_file"]):
      base_columns = ["results_path", "description", "model_name", "train_dir", "test_dir", "val_dir", "epochs", "batch_size", "seed", "model_layers", "model_input_shape", "loss_function", "best_accuracy", "loss", "best_val_accuracy", "val_loss", "test_accuracy", "test_loss"]
    
    with open(self.general_settings["log_file"], "a", newline='') as file:
      writer = csv.writer(file, delimiter=';')

      if base_columns != None:
        writer.writerow(base_columns)

      max_accuracies = (self.history.history[self.acc_name].index(max(self.history.history[self.acc_name])), self.history.history[self.acc_val_name].index(max(self.history.history[self.acc_val_name])))
      writer.writerow(
        [
          results_path,
          self.c_label,
          self.model_name, 
          self.train_dir, 
          self.test_dir, 
          self.val_dir, 
          self.epochs, 
          self.batch_size, 
          self.seed, 
          f"{self.model_name}_summary.txt", 
          str(self.history.model.input_shape), 
          self.history.model.loss, 
          round(self.history.history[self.acc_name][max_accuracies[0]], 3), 
          round(self.history.history[self.loss_name][max_accuracies[0]], 3), 
          round(self.history.history[self.acc_val_name][max_accuracies[1]], 3), 
          round(self.history.history[self.loss_val_name][max_accuracies[1]], 3), 
          round(evaluation[1], 3), 
          round(evaluation[0], 3)]
        )
      print("Created log file")