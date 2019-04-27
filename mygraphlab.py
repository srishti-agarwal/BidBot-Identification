import graphlab as gl
import pandas as pd
import matplotlib.pyplot as plt;  plt.rcdefaults()
class Data():
    def __init__(self):
        # self.bid_data = gl.SFrame.read_csv('data/bids.csv')
        # # self.bid_data['time'] = self.bid_data['time'] / 100000000000
        # # self.train_data = gl.SFrame.read_csv('data/train.csv')
        # # train_bids =  self.bid_data .join(self.train_data, how='left', on='bidder_id')
        # # train_bids.show()
        # # self.train_data = gl.SFrame.read_csv('data/train.csv')
        # train_features = gl.SFrame.read_csv('data/features.csv')
        # train_features_1 =  train_features.join(self.bid_data, how='left', on='bidder_id')
        # # train_features_1.show()
        bid_data = pd.read_csv('data/bids.csv')
        feature_data  = pd.read_csv('data/features.csv')
        all = pd.merge(bid_data,feature_data,how='left',on='bidder_id')
        plt.bar(all['merchandise'], all['nb0fBids'])
        plt.show()
        plt.show()
        # train_features = gl.SFrame(train_features)



Data()