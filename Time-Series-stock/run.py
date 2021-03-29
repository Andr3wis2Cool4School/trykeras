import os 
import json
import math
import numpy as np
import pandas as pd 
from core.data_processor import DataLoader
from core.model import Model
from tensorflow.keras.utils.vis_utils import plot_model
# from tensorflow.keras.utils import plot_model


def main():
    #load config json file
    configs = json.load(open('config.json', 'r'))
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    model = Model()
    mymodel = model.build_model(configs)

    plot_model(mymodel, to_file='model.png', show_shapes=True)






if __name__ == '__main__':
    main()
    

