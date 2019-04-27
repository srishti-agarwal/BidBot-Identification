def mergingFeature(line, features, dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return features[line['bidder_id']]

def findAuctionFeatures(data):
    bid_data = data.bidData
    train_data = data.trainData
    test_data = data.testData
    bidderList = bid_data['bidder_id'].unique()

    # auctionList = bid_data['auction'].unique()
    # grCountryCount = bid_data['country'].groupby(bid_data['auction']).count()
    # train_data['nbCountryPa'] = grCountryCount
    # print (train_data)

    temp = bid_data
    temp['count'] = 1
    auctionData = temp.groupby(['bidder_id','auction'])['count'].sum().reset_index(name='count')
    auctionData = auctionData.pivot(index='bidder_id', columns='auction', values='count')
    meanAucCount = {}
    stdAucCount = {}
    for bidder in bidderList:
        if (auctionData.loc[bidder].mean() > 0):
            meanAucCount[bidder] = auctionData.loc[bidder].mean()
            stdAucCount[bidder] = auctionData.loc[bidder].std()
        else:
            meanAucCount[bidder] = 0
            stdAucCount[bidder] = 0
    train_data['meanBidsPerAuction'] = train_data.apply(lambda x: mergingFeature(x, meanAucCount, bidderList), axis=1)
    test_data['meanBidsPerAuction'] = test_data.apply(lambda x: mergingFeature(x, meanAucCount, bidderList), axis=1)

    train_data['stdBidsPerAuction'] = train_data.apply(lambda x: mergingFeature(x, stdAucCount, bidderList), axis=1)
    test_data['stdBidsPerAuction'] = test_data.apply(lambda x: mergingFeature(x, stdAucCount, bidderList), axis=1)
    return train_data, test_data