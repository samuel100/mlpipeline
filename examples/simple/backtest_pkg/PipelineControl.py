# run data capture
# run ml pipeline
# consume served model
# evaluate post-facto

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from . import MLPipeline as mlp

class PipelineControl():

    data_path = None
    prediction_path = None
    tracking_error = None
    tracking_error_burnin = 0
    sliding_window_size = 0
    retraining_flag = True

    def __init__(self, data_path, prediction_path, tracking_error_burnin=1000, sliding_window_size=0, retraining_flag=True):
        self.data_path = data_path
        self.prediction_path = prediction_path
        self.tracking_error = []
        self.tracking_error_burnin = tracking_error_burnin
        self.sliding_window_size = sliding_window_size
        self.retraining_flag = retraining_flag
    
    def __tracking_error(self, actual, predicted):
        self.tracking_error.append(actual-predicted)
    
    def runPipeline(self):

        f = open(self.prediction_path, 'w')
        print('p_date, prediction', file=f)
        data = pd.read_csv(self.data_path)
        n = len(data)
        mlpipe = mlp.MLPipeline()

        # run an evaluation
        today = datetime.strptime(data['date'][0], '%Y-%m-%d')
        tomorrow = today+timedelta(days=1)
        p = mlpipe.model_consume(data.loc[0, :])

        for i in range(1, n):
            if i > self.sliding_window_size:
                # calculate tracking error
                stdev_tracking_error = 0
                if i > self.tracking_error_burnin: 
                    self.__tracking_error(data['target'][i], p)
                    stdev_tracking_error = np.std(self.tracking_error)

                today = datetime.strptime(data['date'][i], '%Y-%m-%d')
                tomorrow = today+timedelta(days=1)

                if (self.retraining_flag == False) & (i == self.tracking_error_burnin):
                    new_model = mlpipe.model_build(data.loc[0:i, :])
                    mlpipe.model_serve(new_model)
                    print('model built')

                # if we have our retrain cadence e.g. monthly
                if (today.month != tomorrow.month) & (self.retraining_flag) :
                    print(f'{today} rebuilding model....')
                    if self.sliding_window_size > 0:
                        new_model = mlpipe.model_build(data.loc[(i-self.sliding_window_size):i, :])
                    else:
                        new_model = mlpipe.model_build(data.loc[0:i, :])
                    mlpipe.model_serve(new_model)
                    print('model built')
                
                p = mlpipe.model_consume(data.loc[i, :])
                print(f'{tomorrow.date()}, {p}', file=f)
                print(f'tdate={today.date()}, pdate={tomorrow.date()}, prediction={p}, tracking_error={stdev_tracking_error}')
        
        f.close()

