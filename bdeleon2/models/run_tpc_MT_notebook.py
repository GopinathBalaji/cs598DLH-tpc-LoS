from eICU_preprocessing.split_train_test import create_folder
from torch.optim import Adam
from models.tpc_model import TempPointConv
from models.experiment_template_notebook import ExperimentTemplate
from models.initialise_arguments_bdeleon2 import initialise_tpc_arguments
from models.final_experiment_scripts.best_hyperparameters import best_tpc

class TPC(ExperimentTemplate):
    def __init__(self,config,
              n_epochs,
              name,
              base_dir,
              explogger_kwargs,sample=0):
        super().__init__(config=config,
              n_epochs=n_epochs,
              name=name,
              base_dir=base_dir,
              explogger_kwargs=explogger_kwargs)
        self.sample = sample
    def setup(self):
        self.setup_template(sample=self.sample)

        self.model = TempPointConv(config=self.config,
                                   F=self.test_datareader.F,
                                   D=self.test_datareader.D,
                                   no_flat_features=self.test_datareader.no_flat_features).to(device=self.device)
        # self.elog.print(self.model)
        self.optimiser = Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.L2_regularisation)
        self.load_checkpoint("tpc_mimic_mt_model")
        return
        
def loadRunTPCMIMICTestMT():
    c = initialise_tpc_arguments()
    c['exp_name'] = 'TPC'
    c['batch_size'] = 1
    c['save_results_csv'] = True
    c['dataset'] = 'MIMIC'
    c['task'] = 'multitask'
    c=best_tpc(c)
    c['batch_size_test'] = 1
    sample = c['sample']
    log_folder_path = create_folder('models/experiments/{}/{}'.format(c.dataset, c.task), c.exp_name)
    tpc = TPC(config=c,
              n_epochs=c.n_epochs,
              name=c.exp_name,
              base_dir=log_folder_path,
              explogger_kwargs={'folder_format': ''},sample=sample)
 
    tpc.run_test()

if __name__=='__main__':
    
    loadRunTPCMIMICTestMT()
