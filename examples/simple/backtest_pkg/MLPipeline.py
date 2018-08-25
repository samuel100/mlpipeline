# feature engineer on snapshot
# feature selection
# ml build
# serve

from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
import numpy as np

class MLPipeline():

    # we serve the model onto a simple list
    pipeline_stack = None

    def __init__(self):
        # initialize
        self.pipeline_stack = []
    
    def model_build(self, dataset):
        # build a model
        n = len(dataset)
        X = dataset.loc[:,'input1':'input4'][0:(n-1)]
        y = dataset.loc[:,'target'][1:n]

        feature_filter = SelectKBest(f_regression, k=2)
        model = linear_model.LinearRegression(fit_intercept=False)
        my_pipeline = Pipeline([('feature_selection',feature_filter), ('linear_model', model)])
        fitted_pipeline = my_pipeline.fit(X,y)

        return fitted_pipeline
        

    def model_serve(self, model):
        # serve the model (in this case onto a stack)
        self.pipeline_stack.append(model)
    
    def model_consume(self, datapoint):
        # deal with no model in the stack by returning 0
        if len(self.pipeline_stack) == 0:
            return 0
        latest_pipeline = self.pipeline_stack[-1]
        p = np.round(latest_pipeline.predict([datapoint.loc['input1':'input4']])[0])

        return p
    
