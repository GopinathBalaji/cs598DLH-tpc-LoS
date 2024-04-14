from eICU_preprocessing.split_train_test import create_folder
from torch.optim import Adam
from models.transformer_model import Transformer
from models.experiment_template_bdeleon2 import ExperimentTemplate
from models.initialise_arguments_bdeleon2 import initialise_transformer_argumentsMIMIC


class BaselineTransformer(ExperimentTemplate):
    def setup(self):
        self.setup_template()
        self.model = Transformer(config=self.config,
                                 F=self.test_datareader.F,
                                 D=self.test_datareader.D,
                                 no_flat_features=self.test_datareader.no_flat_features,
                                 device=self.device).to(device=self.device)
        self.elog.print(self.model)
        self.optimiser = Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.L2_regularisation)
        self.load_checkpoint("transformer_mimic_model")
        return

def loadRunTransformerMIMICTest():
    c = initialise_transformer_argumentsMIMIC()
    c['exp_name'] = 'Transformer'

    log_folder_path = create_folder('models/experiments/{}/{}'.format(c.dataset, c.task), c.exp_name)
    baseline_transformer = BaselineTransformer(config=c,
                                               n_epochs=c.n_epochs,
                                               name=c.exp_name,
                                               base_dir=log_folder_path,
                                               explogger_kwargs={'folder_format': ''})
 
    baseline_transformer.run_test()

if __name__=='__main__':    
    loadRunTransformerMIMICTest()