from eICU_preprocessing.split_train_test import create_folder
from torch.optim import Adam
from models.lstm_model import BaseLSTM
from models.experiment_template_bdeleon2 import ExperimentTemplate
from models.initialise_arguments import initialise_lstm_arguments
from models.initialise_arguments_bdeleon2 import initialise_lstm_argumentsMIMIC
import torch


class BaselineLSTM(ExperimentTemplate):
    def setup(self):
        self.setup_template()
        self.model = BaseLSTM(config=self.config,
                              F=self.test_datareader.F,
                              D=self.test_datareader.D,
                              no_flat_features=self.test_datareader.no_flat_features).to(device=self.device)
        self.elog.print(self.model)
        self.optimiser = Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.L2_regularisation)
        self.load_checkpoint("lstm_mimic_model")
        return

def loadRunLSTMMIMICTest():
    c = initialise_lstm_argumentsMIMIC()
    c['exp_name'] = 'LSTM'

    log_folder_path = create_folder('models\\experiments\\{}\\{}'.format(c.dataset, c.task), c.exp_name)
    baseline_lstm = BaselineLSTM(config=c,
                                 n_epochs=c.n_epochs,
                                 name=c.exp_name,
                                 base_dir=log_folder_path,
                                 explogger_kwargs={'folder_format': ''})
 
    # baseline_lstm.load_checkpoint("lstm_mimic_model")
    baseline_lstm.run_test()

if __name__=='__main__':

    loadRunLSTMMIMICTest()
