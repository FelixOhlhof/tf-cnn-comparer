import argparse
import configparser
# Parameters
parser = argparse.ArgumentParser(description='Client for tensorflow distributed training')
parser.add_argument("-i", "--id", help="The id of the client", type=int, required=True)
args = parser.parse_args()
config = configparser.ConfigParser()
config.read("config.ini")
from time import sleep
import pickle, sys, os
import socket, util
import time, datetime
import sqlite3
from classifier import Classifier
from binary_classifier import BinaryClassifier
from http import client
from pathlib import Path


class Client():
    def __init__(self):
        self.use_gpu = config.getboolean("CLIENT","USE_GPU")
        if(not self.use_gpu):
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # nessesary for the seed
        self.seed = (int)(config["CLIENT"]["SEED"]) # get the seed
        self.epochs=(int)(config["CLIENT"]["EPOCHS"]) # Number of epoches to be trained
        self.batch_size = (int)(config["CLIENT"]["BATCH_SIZE"])
        self.dataset_name = config["CLIENT"]["DATASET_NAME"]
        self.save_checkpoint = config.getboolean("CLIENT","SAVE_CHECKPOINT")
        self.load_checkpoint = config.getboolean("CLIENT","LOAD_CHECKPOINT")
        self.debug_mode = config.getboolean("CLIENT","DEBUG_MODE")
        self.db = r"results\tf_dist_train_results.db"
        self.train_dir = util.copy_pictures(f"{Path.home()}\\.keras\\datasets\\{self.dataset_name}", self.client_id, self.client_count, self.one_vs_rest, self.one_vs_one, self.debug_mode)
        self.val_dir = config["CLIENT"]["VALIDATION_DATA_DIR"]
        self.test_id = self.get_test_id()
        self.strategy = self.get_strategy()

    def start_dist_training(self):
        self.train_multi_class()
        
    def train_one_vs_rest(self):
        classifier = BinaryClassifier(self.train_dir, self.seed, self.client_id)

        start_time = time.time()
        history = classifier.train_epoch(self.epochs)

        # report
        end_time = time.time()
        time_lapsed = end_time - start_time
        print(classifier.model.history.history)
        model_path = classifier.save_model(self.test_id, self.client_id, self.strategy)
        roc_curve = classifier.save_roc_curve(self.test_id, self.client_id, self.strategy)
        val_loss_curve = util.save_validation_loss_plot(history.history, self.epochs, self.test_id, self.client_id, self.strategy)
        self.report(classifier, time_lapsed, [roc_curve, val_loss_curve])
        self.report_model_db(model_path, classifier.class_names[0], classifier.class_names[1])


    def train_multi_class(self):
        best_accuracy = 0.0
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        classifier = Classifier(self.client_id, self.client_count, self.train_dir, self.val_dir, self.seed, self.save_checkpoint, self.load_checkpoint, self.batch_size, self.shuffle_data_mode)

        start_time = time.time()

        for i in range(self.epochs):
            print(f"************* EPOCH {i+1} *************")
            if(self.shuffle_data_mode):
                hist = classifier.train_epoch_in_shuffle_mode(inner_epoch=self.inner_epochs, current_epoch=i)
            else:
                hist = classifier.train(inner_epoch=self.inner_epochs)

            history = self.updateHistory(hist, history)
            new_accuracy = hist.history['val_accuracy'][0]
            print(hist.history)
            
            if(self.client_count != 1):
                if new_accuracy > best_accuracy or self.send_weights_without_improvement:
                    print(f" -> best_accuracy: {best_accuracy} new_accuracy: {new_accuracy}")
                    # send weights and get new weights
                    updated_weights = self.retrieve_new_weights(old_weights=classifier.model.get_weights())
                else:
                    # send skip to server
                    print(f"Accuracy did not improve -> best_accuracy: {best_accuracy} new_accuracy: {new_accuracy}")
                    updated_weights = self.send_skip(old_weights=classifier.model.get_weights())

                # set updated weights    
                classifier.model.set_weights(updated_weights)

            #if(new_accuracy > best_accuracy):
            best_accuracy = new_accuracy
            print()

        print(classifier.model.history.history)
        #score = classifier.calculate_score() not in use anymore

        # report
        end_time = time.time()
        time_lapsed = end_time - start_time
        graph_path = util.save_validation_loss_plot(history, self.epochs, self.test_id, self.client_id, self.strategy)
        accuracy, loss, val_accuracy, val_loss, evaluation_time = classifier.evaluate(self.client_id)

        self.report(classifier, time_lapsed, graph_path, evaluation_time, accuracy, loss, val_accuracy, val_loss)
        model_path = classifier.save_model(self.test_id, self.client_id, self.strategy)
        self.report_model_db(model_path, "", "")

    def report(self, classifier, time_lapsed, graph_path, evaluation_time = 0, accuracy = 0, loss = 0, val_accuracy = 0, val_loss = 0):
        if(self.client_id == 1):
            self.report_results_db(classifier, time_lapsed, evaluation_time, accuracy, loss, val_accuracy, val_loss)
            
        while(self.get_test_id() == self.test_id): # checking if worker1 already inserted the result
            print("Waiting for worker1 to insert result into db...")
            sleep(0.5)

        if isinstance(graph_path, list):
            for g in graph_path:
                self.report_graph_db(g)
        else:
            self.report_graph_db(graph_path)


    def report_results_db(self, classifier, time_lapsed, evaluation_time, accuracy, loss, val_accuracy, val_loss):
        con = sqlite3.connect(self.db)
        cur = con.cursor()  
        cur.execute(f"INSERT INTO Results (TEST_ID, DATASET_NAME, NUMBER_OF_WORKERS, EPOCHS, INNER_EPOCHS, BATCH_SIZE_PER_WORKER, STRATEGY, SHUFFLE_DATA, SEND_WEIGHTS_WITHOUT_IMPROVEMENT, USE_GPU, SEED, TOTAL_TRAINING_TIME, EVALUATION_TIME, TIMESTAMP, ACCURACY, LOSS, VAL_ACCURACY, VAL_LOSS, ACTIVE) VALUES ({self.test_id}, '{self.dataset_name}', {self.client_count}, {self.epochs}, {self.inner_epochs}, {classifier.batch_size}, '{self.strategy}', {(int)(self.shuffle_data_mode)}, {(int)(self.send_weights_without_improvement)}, {(int)(self.use_gpu)}, '{self.seed}', '{util.time_convert(time_lapsed)}', {round(evaluation_time, 2)}, '{datetime.datetime.now()}', {round(accuracy, 4)}, {round(loss, 4)}, {round(val_accuracy, 4)}, {round(val_loss, 4)}, -1);")
        con.commit()
        con.close()
        print("Inserted result into db")

    def report_graph_db(self, graph_path):
        con = sqlite3.connect(self.db)
        cur = con.cursor()    
        cur.execute(f"INSERT INTO Graphs (TEST_ID, WORKER_ID, PATH) VALUES ({self.test_id}, {self.client_id}, '{graph_path}');")
        con.commit()
        con.close()
        print("Inserted graph into db")

    def report_model_db(self, model_path, class_1, class_2):
        con = sqlite3.connect(self.db)
        cur = con.cursor()    
        cur.execute(f"INSERT INTO Models (TEST_ID, MODEL_PATH, STRATEGY, CLASS_1, CLASS_2, WORKER_ID) VALUES ({self.test_id}, '{model_path}', '{self.strategy}', '{class_1}', '{class_2}', {self.client_id});")
        con.commit()
        con.close()
        print("Inserted model into db")

    def get_strategy(self):
        if(self.one_vs_one):
            return "ONE_VS_ONE"
        elif(self.one_vs_rest):
            return "ONE_VS_REST"
        else:
            return "MULTI_CLASS"

    def get_test_id(self):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        test_id = cur.execute(f"SELECT Count(*) from Results;").fetchone()[0] + 1
        con.close()
        return test_id

    def send_skip(self, old_weights):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.hostname, self.port))

            #send skip flag
            s.send(pickle.dumps(-1))
            print(pickle.loads(s.recv(1024)))

            #get size of new weights
            msg_len = pickle.loads(s.recv(1024))
            print(f"Recieved size from Server {msg_len}")

            if(msg_len == -1):
                # no worker improved, server sent skip flag
                print("No worker improved, continuing with old weights!")
                return old_weights

            # avg weights from other workers
            new_weights = pickle.loads(s.recv(msg_len))
            print("Recieved new weights")
            return new_weights


    def retrieve_new_weights(self, old_weights):
        print("First weight before merging: ", old_weights[0][0][0][0][0])
        b_weights = pickle.dumps(old_weights)
        print(sys.getsizeof(b_weights), " bytes will be transfered to server...")

        # Send weights and get new new calculated weights
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.hostname, self.port))
            
            #send size of weights
            s.send(pickle.dumps(sys.getsizeof(b_weights)))
            print(pickle.loads(s.recv(1024)))

            #send weights
            s.sendall(b_weights)
            #print(pickle.loads(s.recv(1024)))

            #get size of new weights
            msg_len = pickle.loads(s.recv(1024))

            #rec new weights
            print(f"Recieved length of new weights {msg_len}")
            data = s.recv(msg_len)

            new_weights = pickle.loads(data)
            print("Recieved new weights! First weight after merging:", new_weights[0][0][0][0][0])
            return new_weights

    def updateHistory(self, hist, history):
        history['loss'].append(hist.history['loss'][0])
        history['accuracy'].append(hist.history['accuracy'][0])
        history['val_loss'].append(hist.history['val_loss'][0])
        history['val_accuracy'].append(hist.history['val_accuracy'][0])
        return history

if __name__ == "__main__":
    client = Client()    
    client.start_dist_training()