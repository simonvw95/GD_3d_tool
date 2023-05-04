import random

import constants
import visualization_tool
import pyqtgraph as pg

if __name__ == '__main__':
    #randomize evaluation set
    #evaluation_set = constants.evaluation_set[1:]
    #random.shuffle(evaluation_set)
    #constants.evaluation_set[1:] = evaluation_set

    #start tool
    win = visualization_tool.Tool()
    win.showMaximized()
    pg.exec()