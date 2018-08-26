# The importance of backtesting ML pipelines

Often Machine Learning (ML) literature shows the *Training-Validate-Serve* (TVS) pattern where we train and validate our model before serving the model so that predictions can be made on new data arriving into the system. The *snapshots* of our data taken through time would look as follows:

<img src="doc/imgs/t-v-s.png?raw=true" alt="tvs" width="400px"/>

The diagram shows that as our dataset grows through time we are not refitting our ML pipeline on newer data but instead making predictions (inferencing) based on a model trained from an increasingly distant past. The TVS pattern is a good approach if the underlying process we are modelling is *fixed* (does not vary though time) i.e.  

$y=f(\mathbf{x})+r$

where $y$ is the target (or response) variable we want to predict, $\mathbf{x}$ is a vector containing the input (or predictor) variables, $f$ is the underlying function that maps the input variables to the target variable (this is the function that a machine learning algorithm e.g. neural network will learn) and $r$ is the unseen error.

However, for most real-world applications (e.g. fraud prediction, loan default prediction, stock price forecasting) the underlying process will *vary* through time i.e.

$$y=f_t(\mathbf{x}) + r$$

where the underlying function $f$ we need to model using an ML algorithm has a time dependency. In such a case, using the TVS pattern will lead to our predictive model becoming *stale* as new data is collected and we will observe large *tracking error* (the difference between our in-sample error and out-of-sample error).

The solution to the stale model problem is to periodically refit a machine learning pipeline (feature selection, training, business rules) as new data arrives. The refitting process can either be scheduled to occur based on time (e.g. the end of every month) or when the observed error of the model reaches a pre-determined level of unacceptability. The aim of this document is to provide guidance on how to refit machine learning pipelines by leveraging a *backtesting* paradigm. 

Backtesting is the process of applying an analytical method to historical snapshots of data to see how accurately the strategy or method would have predicted actual results. The benefits of a well engineered backtesting framework include:

* retraining and tracking error is implicit
* reduces probability of *data leakage* i.e. where we use data that would not be available at training time to make a prediction
* measures impact of any business rules or parameter changes in the model
* it can be leveraged to produce ongoing predictions on as yet unseen data i.e. a production system. This has the benefit that the production system is the same as the ML development system.

In addition, the approach we are recommending allows for:

* integration of cloud services to provide better scaling
    - training on GPU, Spark
    - serving and consuming models through REST API endpoints
* replicability through docker containers
* integration with auditing tools e.g. MLflow
* integration with continuous integration tools e.g. VSTS, Jenkins.

Often in machine learning projects retraining is an afterthought once the model has been built using the TVS pattern. We therefore strongly advise implementing a backtesting framework at the start of a project to:

* leverage the benefits above
* reduce experimentation-to-production time
 

## Data Snapshotting methods

When we are creating an ML pipeline we are doing this on a snapshot of data at a particular moment in time. The goal of backtesting is to repeatedly present to the ML pipeline a data snapshots using our historical data and refit the pipeline for each snapshot i.e. 

<img src="doc/imgs/increasing.png?raw=true" alt="tvs" width="400px"/>

Here we see that we are refitting the ML pipeline on newer data, but we are using *all* the available data in the snapshot. In a process where the underlying process varies through time, this method can be slow to capture the changes in the model. A solution to this problem is to use a *sliding window* data snapshot methodology i.e.

<img src="doc/imgs/sliding.png?raw=true" alt="tvs" width="400px"/>

Here we see that whilst we start off using all the available data in the first snapshot as time moves in the backtest on we are using the most recent $n$ number of data points to fit our ML pipeline to.

Later in this article we will use these different snapshot methods on synthetic data to better demonstrate which method to use.

## Beware of Parameter Shift

Whilst a backtesting framework can reduce data leakage by ensuring that we are retraining our ML pipelines on data available at the time (i.e. no peeking ahead), care is needed with *parameters*. 

For example, in the scenario of forecasting credit risk (i.e. whether to give a person credit or not) we use a threshold of *p*% probability from our ML model to determine that a person will default. In our backtest framework we run multiple backtests to determine that a threshold of 80% provides the best results. It is tempting to take this observation as *the truth* and that going forward in our real-time production system the probability threshold of 80% will give the best results. This is not a true statement.

In most ML systems we will witness something called *parameter shift* - this is where a parameter that produces the best in-sample results does not produce the best results on unseen data. In our credit risk scenario, it could be that when we put the ML system into production we witness that a probability threshold of 70% provides the best results. 

Therefore, to protect your ML system from parameter shift you should seek to use an *analytical method* at retraining time to determine the parameter. For example, in the credit risk scenario above we could use a grid-search at training time on the *p*% probability parameter to optimize the F1 score and use this on the un-seen data. Backtesting provides us with a framework with which to test these analytical methods.

## Designing a backtesting framework

The code we have provided consists of 3 classes

### DataCapture

### MLPipeline

### PipelineControl

## Experiments

Take a look at [this](experiments.ipynb) notebook in the repo that takes you through running the backtesting framework on fixed and varying processes.

