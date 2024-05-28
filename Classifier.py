import tensorflow as tf
import numpy as np
import random as rn
import models
import matplotlib.pyplot as plt
import csv
import os


from tensorflow import keras
from keras import layers
from keras.models import Sequential
from sklearn.metrics import roc_curve, auc


class Classifier():
  def __init__(self, train_dir, test_dir, val_dir, seed, batch_size, model_name, img_height, img_width, epochs):
    self.classes = ["morph", "no_morph"]
    np.random.seed(37)
    rn.seed(seed)
    tf.random.set_seed(seed)
    self.seed = seed
    self.batch_size = batch_size
    self.img_height = img_height
    self.img_width = img_width
    self.callbacks_list = None
    self.epochs = epochs
    self.model_name = model_name
    model_method = getattr(models, model_name)
    self.model = model_method(img_height, img_width)
    self.train_dir = train_dir
    self.val_dir = val_dir
    self.test_dir = test_dir
    self.train_ds, self.val_ds, self.test_ds = self.get_datasets()

  def get_datasets(self):
    train_ds = tf.keras.utils.image_dataset_from_directory(
      self.train_dir,
      seed=self.seed,
      label_mode="binary",
      image_size=(self.img_height, self.img_width),
      batch_size=self.batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
      self.val_dir,
      seed=self.seed,
      label_mode="binary",
      image_size=(self.img_height, self.img_width),
      batch_size=self.batch_size)
    test_ds = tf.keras.utils.image_dataset_from_directory(
      self.test_dir,
      seed=self.seed,
      label_mode="binary",
      image_size=(self.img_height, self.img_width),
      batch_size=self.batch_size)
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

  def save_validation_loss_plot(self, path):
      #visualize training results
      acc = self.history.history[self.acc_name]
      val_acc = self.history.history[self.acc_val_name]

      loss = self.history.history[self.loss_name]
      val_loss = self.history.history[self.loss_val_name]

      epochs_range = range(self.epochs)

      plt.figure(figsize=(8, 8))
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
      plt.savefig(os.path.join(path, f"{self.model_name}_val_loss.png"), dpi=600)
      plt.close()

  def save_model(self, path):
    self.model.save(os.path.join(path, f"{self.model_name}_model.h5"))
  
  def write_report(self, path):
    evaluation = self.model.evaluate(self.test_ds)
    file_name = os.path.join(path, "log.csv")
    base_columns = None

    if not os.path.isfile(file_name):
      base_columns = ["model_name", "train_dir", "test_dir", "val_dir", "epochs", "batch_size", "seed", "model_layers", "model_input_shape", "loss_function", "best_accuracy", "loss", "best_val_accuracy", "val_loss", "test_accuracy", "test_loss"]
    
    with open(file_name, "a", newline='') as file:
      writer = csv.writer(file, delimiter=';')

      if base_columns != None:
        writer.writerow(base_columns)

      max_accuracies = (self.history.history[self.acc_name].index(max(self.history.history[self.acc_name])), self.history.history[self.acc_val_name].index(max(self.history.history[self.acc_val_name])))
      writer.writerow(
        [self.model_name, self.train_dir, self.test_dir, self.val_dir, self.epochs, self.batch_size, self.seed, str(self.history.model.layers), str(self.history.model.input_shape), self.history.model.loss, self.history.history[self.acc_name][max_accuracies[0]], list(self.history.history)[1][max_accuracies[0]], self.history.history[self.loss_name][max_accuracies[1]], self.history.history[self.loss_val_name][max_accuracies[1]], evaluation[1], evaluation[0]]
        )
      print("Created log file")