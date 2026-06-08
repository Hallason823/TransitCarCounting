import os
import matplotlib.pyplot as plt
import seaborn as sns

class PlotManager:
    """Saves training curves and confusion matrices to disk."""

    def __init__(self, results_dir='src/plot/results_images/', cm_dir='src/plot/confusion_matrix_images/'):
        self.results_dir = results_dir
        self.cm_dir = cm_dir
        self.plot_count = 0

    def initializePlotDirs(self):
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.cm_dir, exist_ok=True)

    def plotLosses(self, history, loss_name='Cross-Entropy Loss'):
        self.initializePlotDirs()
        self.plot_count += 1
        epochs = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(epochs, history['train_loss'], label='Train')
        axes[0].plot(epochs, history['val_loss'],   label='Validation')
        axes[0].set_title(loss_name)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[1].plot(epochs, [a * 100 for a in history['train_acc']], label='Train')
        axes[1].plot(epochs, [a * 100 for a in history['val_acc']],   label='Validation')
        axes[1].set_title('Accuracy (%)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        plt.tight_layout()
        path = os.path.join(self.results_dir, f'training_{self.plot_count}.png')
        plt.savefig(path)
        plt.close()
        print(f'Training curves saved: {path}')

    def plotConfusionMatrix(self, categories_name, confusion_mat):
        self.plot_count += 1
        size = max(8, len(categories_name))
        plt.figure(figsize=(size, size - 1))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=categories_name, yticklabels=categories_name)
        plt.title('Confusion Matrix — Cars per Image')
        plt.ylabel('True Count')
        plt.xlabel('Predicted Count')
        plt.tight_layout()
        path = os.path.join(self.cm_dir, f'confusion_matrix_{self.plot_count}.png')
        plt.savefig(path)
        plt.close()
        print(f'Confusion matrix saved: {path}')

    def saveAllImages(self):
        print(f'\nAll plots saved to {self.results_dir} and {self.cm_dir}')