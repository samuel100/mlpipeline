# feature engineer on snapshot
# feature selection
# ml build
# serve

from sklearn import linear_model
import numpy as np

class MLPipeline():

    # we serve the model onto a simple list
    pipeline_stack = None

    def __init__(self):
        # initialize
        self.pipeline_stack = []
    
    def model_build(self, dataset):
        # build a model
        model = linear_model.LinearRegression(fit_intercept=False)
        X = dataset.loc[:,'input1':'input2']
        y = dataset.loc[:,'target']
        n = X.shape[0]
        model.fit(X[0:(n-1)], y[1:n])

        return model
        

    def model_serve(self, model):
        # serve the model (in this case onto a stack)
        self.pipeline_stack.append(model)
    
    def model_consume(self, datapoint):
        # deal with no model in the stack by returning 0
        if len(self.pipeline_stack) == 0:
            return 0
        latest_model = self.pipeline_stack[-1]
        p = np.round(latest_model.predict([datapoint.loc['input1':'input2']])[0])

        return p
    
