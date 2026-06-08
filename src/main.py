import sys
import numpy as np
from config.config import *
from taskManager import taskManager
from plot.plot import PlotManager

if __name__ == '__main__':
    params = sys.argv

    if '-h' in params or '--help' in params:
        printHelp()
        exit(0)

    if len(params) < 2:
        print('\nProvide a model name: RESNET | EFFICIENTNET\n')
        printHelp()
        exit(1)

    model  = params[1].upper()
    config = getConfigs(params[2:])
    num_iter = config[0]
    config   = config[1:]
    plot = PlotManager()
    global_evaluation_metrics = []

    for it in range(num_iter):
        print(f'\nIteration {it + 1}\n')
        manager = taskManager(model, config, plot)
        manager.runClassifierBasedOnModel()
        global_evaluation_metrics += manager.local_evaluation_metrics_by_run
        plot = manager.plot
        printError(manager.error)

    manager.plot.saveAllImages()

    if global_evaluation_metrics:
        arr = np.array(global_evaluation_metrics)
        print('\n--- Global Results ---')
        for i, name in enumerate(['Accuracy', 'F1-Score (macro)', 'F1-Score (weighted)']):
            print(f"  {name}: mean={arr[:, i].mean():.4f}  std={arr[:, i].std():.4f}")

    print('\nDone!')
    exit(0)