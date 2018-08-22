import PipelineControl as pc

if __name__ == "__main__" :
    pipe = pc.PipelineControl('data/data.csv', 'data/predictions.csv')
    pipe.runPipeline()

