import pandas as pd


class Data():
    def __init__(self, path):
        self.trainData = pd.read_csv(path + '/train.csv')
        self.testData = pd.read_csv(path + '/test.csv')
        self.bidData = pd.read_csv(path + '/bids.csv')