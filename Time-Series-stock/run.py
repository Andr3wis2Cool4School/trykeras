import os 
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.legend()
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediciton')
    
    plt.savefig('results_multiple.png')


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.savefig('results.png')


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
    print(mymodel.summary())

    """
        Model: "sequential"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        lstm (LSTM)                  (None, 49, 100)           41200
        _________________________________________________________________
        dropout (Dropout)            (None, 49, 100)           0
        _________________________________________________________________
        lstm_1 (LSTM)                (None, 49, 100)           80400
        _________________________________________________________________
        lstm_2 (LSTM)                (None, 100)               80400
        _________________________________________________________________
        dropout_1 (Dropout)          (None, 100)               0
        _________________________________________________________________
        dense (Dense)                (None, 1)                 101
        =================================================================
        Total params: 202,101
        Trainable params: 202,101
        Non-trainable params: 0
    """
#Add the training data
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise'] 
    )
    print(x.shape)
    print(y.shape)

    # Train our model 
    model.train(
        x, 
        y, 
        epochs = configs['training']['epochs'],
        batch_size = configs['training']['batch_size'],
        save_dir = configs['model']['save_dir']
    )

    # Evaluate the performance
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # show
    predicitons_multiseq = model.predict_sequences_multiple(x_test, 
                                                            configs['data']['sequence_length'], 
                                                            configs['data']['sequence_length'])
    predictions_pointbypoint = model.predict_point_by_point(x_test)
    plot_results_multiple(predicitons_multiseq, y_test, configs['data']['sequence_length'])
    plot_results(predictions_pointbypoint, y_test)


if __name__ == '__main__':
    main()
    

