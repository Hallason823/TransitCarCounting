from config.config import *
from data.dataManager import DataManager
from models.evaluators import EvaluatorNetworks
from classifier.classifier import Classifier
from plot.plot import PlotManager

MODEL_INDEX = {'RESNET': 0, 'EFFICIENTNET': 1}

class taskManager:
    def __init__(self, model, config, plot):
        self.model      = model
        self.config     = config
        self.plot       = plot
        self.split_mode = self.config[0]
        self.iter_limit = 1 if self.split_mode == 'Hold-out' else self.config[5]
        self.local_evaluation_metrics_by_run = []
        self.error = False
        self.initializeDataset()

    def initializeDataset(self):
        if not self.error:
            if self.model in MODEL_INDEX:
                self.ds = DataManager(
                    folder_path    = DATA_FOLDER,
                    split_mode     = self.split_mode,
                    weights        = self.config[4],
                    k_groups       = self.config[5],
                    img_size       = IMG_SIZE
                )
            else:
                print(f'\nUnknown model: {self.model}\n')
                self.error = True

    def initializeClassifier(self):
        if not self.error:
            model_idx = MODEL_INDEX[self.model]
            self.evaluator = EvaluatorNetworks(
                model_idx       = model_idx,
                device          = DEVICE,
                train_ds        = self.ds.train_ds,
                val_ds          = self.ds.val_ds,
                num_classes     = self.ds.num_classes,
                learning_rate   = self.config[3],
                n_epochs        = self.config[1],
                batch_size      = self.config[2],
                freeze_backbone = self.config[6]
            )
            self.plot.plotLosses(self.evaluator.history)
            predictions, true_labels = self.evaluator.predictDataset(self.ds.test_ds)
            self.classifier = Classifier(predictions, true_labels, self.ds.categories_name)
            self.plot.plotConfusionMatrix(self.ds.categories_name, self.classifier.confusion_mat)
            self.local_evaluation_metrics_by_run.append(self.classifier.evaluation_metrics)
        else:
            self.error = True

    def runClassifierBasedOnModel(self):
        if not self.error:
            if self.split_mode == 'Hold-out':
                self.initializeClassifier()
            elif self.split_mode == 'K-fold cross-validation':
                self.initializeClassifier()
                if self.ds.k_val < self.iter_limit:
                    self.ds.updateTheDistributionFolds()
                    self.runClassifierBasedOnModel()
            else:
                self.error = True