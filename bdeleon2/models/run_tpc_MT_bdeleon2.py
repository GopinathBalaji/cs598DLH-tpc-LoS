from eICU_preprocessing.split_train_test import create_folder
from torch.optim import Adam
from models.tpc_model import TempPointConv
from models.experiment_template_bdeleon2 import ExperimentTemplate
from models.initialise_arguments_bdeleon2 import initialise_tpc_argumentsMIMICMT
import torch
from models.final_experiment_scripts.best_hyperparameters import best_tpc

class TPC(ExperimentTemplate):
    def setup(self):
        self.setup_template()

        self.model = TempPointConv(config=self.config,
                                   F=self.test_datareader.F,
                                   D=self.test_datareader.D,
                                   no_flat_features=self.test_datareader.no_flat_features).to(device=self.device)
        self.elog.print(self.model)
        self.optimiser = Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.L2_regularisation)
        self.load_checkpoint("tpc_mimic_mt_model")
        return

def loadRunTPCMIMICTestMT():
    c = initialise_tpc_argumentsMIMICMT()
    c['exp_name'] = 'TPC'
    c=best_tpc(c)

    log_folder_path = create_folder('models/experiments/{}/{}'.format(c.dataset, c.task), c.exp_name)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': ''})
 
    tpc.run_test()

if __name__=='__main__':
    loadRunTPCMIMICTestMT()
