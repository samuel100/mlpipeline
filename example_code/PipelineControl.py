# run data capture
# run ml pipeline
# consume served model
# evaluate post-facto

import pandas as pd
from datetime import datetime, timedelta
import MLPipeline as mlp

class PipelineControl():

    data_path = None
    prediction_path = None

    def __init__(self, data_path, prediction_path):
        self.data_path = data_path
        self.prediction_path = prediction_path
    
    def runPipeline(self, retrain_cadence='EOM'):
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
            today = datetime.strptime(data['date'][i], '%Y-%m-%d')
            tomorrow = today+timedelta(days=1)

            # if we have our retrain cadence e.g. monthly
            if today.month != tomorrow.month :
                print(f'{today} rebuilding model....')
                new_model = mlpipe.model_build(data.loc[0:i, :])
                mlpipe.model_serve(new_model)
                print('model built')
            
            p = mlpipe.model_consume(data.loc[i, :])
            print(f'{tomorrow.date()}, {p}', file=f)
            print(f'prediction for {tomorrow.date()} is {p}')
        
        f.close()