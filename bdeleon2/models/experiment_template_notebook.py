import torch
from eICU_preprocessing.reader import eICUReader
from MIMIC_preprocessing.reader_notebook import MIMICReader
from eICU_preprocessing.split_train_test import create_folder
import numpy as np
from models.metrics import print_metrics_regression, print_metrics_mortality
from trixi.experiment.pytorchexperiment import PytorchExperiment
import os
from models.shuffle_train import shuffle_train
from eICU_preprocessing.run_all_preprocessing import eICU_path
from MIMIC_preprocessing.run_all_preprocessing import MIMIC_path


# view the results by running: python3 -m trixi.browser --port 8080 BASEDIR

def save_to_csv(PyTorchExperimentLogger, data, path, header=None):
    """
        Saves a numpy array to csv in the experiment save dir

        Args:
            data: The array to be stored as a save file
            path: sub path in the save folder (or simply filename)
    """

    folder_path = create_folder(PyTorchExperimentLogger.save_dir, os.path.dirname(path))
    file_path = folder_path + '/' + os.path.basename(path)
    if not file_path.endswith('.csv'):
        file_path += '.csv'
    np.savetxt(file_path, data, delimiter=',', header=header, comments='')
    return

def remove_padding(y, mask, device):
    """
        Filters out padding from tensor of predictions or labels

        Args:
            y: tensor of los predictions or labels
            mask (bool_type): tensor showing which values are padding (0) and which are data (1)
    """
    # note it's fine to call .cpu() on a tensor already on the cpu
    y = y.where(mask, torch.tensor(float('nan')).to(device=device)).flatten().detach().cpu().numpy()
    y = y[~np.isnan(y)]
    return y


class ExperimentTemplate(PytorchExperiment):

    def setup_template(self,sample=0):

        # self.elog.print("Config:")
        # self.elog.print(self.config)
        if not self.config.disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # set bool type for where statements
        self.bool_type = torch.cuda.BoolTensor if self.device == torch.device('cuda') else torch.BoolTensor

        # get datareader
        if self.config.dataset == 'MIMIC':
            self.datareader = MIMICReader
            self.data_path = MIMIC_path
        
        self.test_datareader = self.datareader(self.data_path + 'test', device=self.device,
                                          labs_only=self.config.labs_only, no_labs=self.config.no_labs,sample=sample)
        self.no_train_batches = 0
        self.checkpoint_counter = 0

        self.model = None
        self.optimiser = None

        # add a new function to elog (will save to csv, rather than as a numpy array like elog.save_numpy_data)
        self.elog.save_to_csv = lambda data, filepath, header: save_to_csv(self.elog, data, filepath, header)
        self.remove_padding = lambda y, mask: remove_padding(y, mask, device=self.device)
        self.elog.print('Experiment set up.')

        return

    def train(self, epoch, mort_pred_time=24):
        return

    def validate(self, epoch, mort_pred_time=24):
        return

    def test(self, mort_pred_time=24):

        self.model.eval()
        test_batches = self.test_datareader.batch_gen(batch_size=self.config.batch_size_test)
        test_loss = []
        test_y_hat_los = np.array([])
        test_y_los = np.array([])
        test_y_hat_mort = np.array([])
        test_y_mort = np.array([])

        for batch in test_batches:

            # unpack batch
            if self.config.dataset == 'MIMIC':
                padded, mask, flat, los_labels, mort_labels, seq_lengths = batch
                diagnoses = None
            else:
                padded, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = batch
            # print("LOS LABELS:",los_labels.shape)
            # print("MORT LABELS",mort_labels.shape)
            # print("SEQ len:",seq_lengths)
            y_hat_los, y_hat_mort = self.model(padded, diagnoses, flat)
            loss = self.model.loss(y_hat_los, y_hat_mort, los_labels, mort_labels, mask, seq_lengths, self.device,
                                   self.config.sum_losses, self.config.loss)
            test_loss.append(loss.item())  # can't add the model.loss directly because it causes a memory leak

            if self.config.task in ('LoS', 'multitask'):
                test_y_hat_los = np.append(test_y_hat_los,
                                          self.remove_padding(y_hat_los, mask.type(self.bool_type)))
                test_y_los = np.append(test_y_los, self.remove_padding(los_labels, mask.type(self.bool_type)))
            if self.config.task in ('mortality', 'multitask') and mort_labels.shape[1] >= mort_pred_time:
                test_y_hat_mort = np.append(test_y_hat_mort,
                                           self.remove_padding(y_hat_mort[:, mort_pred_time],
                                                               mask.type(self.bool_type)[:, mort_pred_time]))
                test_y_mort = np.append(test_y_mort, self.remove_padding(mort_labels[:, mort_pred_time],
                                                                         mask.type(self.bool_type)[:, mort_pred_time]))

        print('Test Metrics:')
        mean_test_loss = sum(test_loss) / len(test_loss)

        # if self.config.task in ('LoS', 'multitask'):
        #     los_metrics_list = print_metrics_regression(test_y_los, test_y_hat_los, elog=self.elog)  # order: mad, mse, mape, msle, r2, kappa
        #     for metric_name, metric in zip(['mad', 'mse', 'mape', 'msle', 'r2', 'kappa'], los_metrics_list):
        #         self.add_result(value=metric, name='test_' + metric_name)
        # if self.config.task in ('mortality', 'multitask'):
            # mort_metrics_list = print_metrics_mortality(test_y_mort, test_y_hat_mort, elog=self.elog)
            # for metric_name, metric in zip(['acc', 'prec0', 'prec1', 'rec0', 'rec1', 'auroc', 'auprc', 'f1macro'],
            #                                mort_metrics_list):
            #     self.add_result(value=metric, name='test_' + metric_name)

        if self.config.save_results_csv:
            if self.config.task in ('LoS', 'multitask'):
                self.elog.save_to_csv(np.vstack((test_y_hat_los, test_y_los)).transpose(), 'val_predictions_los.csv', header='los_predictions, label')
                print("LoS prediction : ground truth")
                res = np.vstack((test_y_hat_los, test_y_los)).transpose()
                # print(len(res))
                # res = res[-5:]
                skp =len(res) // 10
                if skp == 0:
                    skp = 1
                for i in range(0,len(res),skp):
                    print(res[i][0],":",res[i][1])
                # for row in res:
                #     print(row[0],":",row[1])
            if self.config.task in ('mortality', 'multitask'):
                self.elog.save_to_csv(np.vstack((test_y_hat_mort, test_y_mort)).transpose(), 'val_predictions_mort.csv', header='mort_predictions, label')
                print("Mortality prediction : ground truth")
                res = np.vstack((test_y_hat_mort, test_y_mort)).transpose()
                for row in res:
                    print(row[0],":",row[1])
        self.elog.print('Test Loss: {:3.4f}'.format(mean_test_loss))
        return

    def resume(self, epoch):
        return
