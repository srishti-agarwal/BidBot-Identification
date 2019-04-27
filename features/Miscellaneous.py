import pandas as pd
from DataProcess import Data
def mergingFeature(line, features, dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return features[line['bidder_id']]

def findMiscellaneousFeatures(data):
    bid_data = data.bidData
    train_data = data.trainData
    test_data = data.testData
    bidderList = bid_data['bidder_id'].unique()

    meanCtryCount = {}
    stdCtryCount = {}
    meanUrlCount = {}
    stdUrlCount = {}
    meanIpCount = {}
    stdIpCount = {}
    meanDeviceCount = {}
    stdDeviceCount = {}
    for bidder in bidderList:
        meanCtryCount[bidder] = 0
        meanUrlCount[bidder] = 0
        meanIpCount[bidder] = 0
        meanDeviceCount[bidder] = 0
        stdCtryCount[bidder] = 0
        stdDeviceCount[bidder] = 0
        stdUrlCount[bidder] = 0
        stdIpCount[bidder] = 0

    temp = bid_data
    temp['count'] = 1
    countryData = temp.groupby(['bidder_id','country'])['count'].sum().reset_index(name='count')

    countryData = countryData.pivot(index='bidder_id', columns='country', values='count')
    bidderList = countryData.index.values

    for bidder in bidderList:
        if (countryData.loc[bidder].mean() > 0):
            meanCtryCount[bidder] = countryData.loc[bidder].mean()

        if (countryData.loc[bidder].std() > 0):
            stdCtryCount[bidder] = countryData.loc[bidder].std()


    # temp = bid_data

    count_bids = bid_data.groupby('url').bid_id.count().reset_index()
    count_bids = count_bids.rename(columns={'bid_id': 'urlCount'})
    b = pd.merge(bid_data, count_bids[['url', 'urlCount']], on='url', how='left')
    t = pd.merge(train_data, b, on='bidder_id', how='left')
    mt = t.groupby('bidder_id').urlCount.mean().reset_index()
    mt = mt.rename(columns={'urlCount': 'meanUrlCount'})
    st = t.groupby('bidder_id').urlCount.std().reset_index()
    st = st.rename(columns={'urlCount': 'stdUrlCount'})
    t = pd.merge(mt, st, on='bidder_id', how='left')
    # print (t.head(5))
    # print (t.shape)
    # train_data = pd.merge(train_data, t[['bidder_id', 'meanUrlCount','stdUrlCount']], on='bidder_id', how='left')
    t = pd.merge(test_data, b, on='bidder_id', how='left')
    mt = t.groupby('bidder_id').urlCount.mean().reset_index()
    mt = mt.rename(columns={'urlCount': 'meanUrlCount'})
    st = t.groupby('bidder_id').urlCount.std().reset_index()
    st = st.rename(columns={'urlCount': 'stdUrlCount'})
    tt = pd.merge(mt, st, on='bidder_id', how='left')
    # test_data = pd.merge(test_data, tt[['bidder_id', 'meanUrlCount', 'stdUrlCount']], on='bidder_id', how='left')


    count_bids = bid_data.groupby('ip').bid_id.count().reset_index()
    count_bids = count_bids.rename(columns={'bid_id': 'ipCount'})
    b = pd.merge(bid_data, count_bids[['ip', 'ipCount']], on='ip', how='left')
    t = pd.merge(train_data, b, on='bidder_id', how='left')
    mt = t.groupby('bidder_id').ipCount.mean().reset_index()
    mt = mt.rename(columns={'ipCount': 'meanIpCount'})
    st = t.groupby('bidder_id').ipCount.std().reset_index()
    st = st.rename(columns={'ipCount': 'stdIpCount'})
    t = pd.merge(mt, st, on='bidder_id', how='left')
    # train_data = pd.merge(train_data, t[['bidder_id', 'meanIpCount', 'stdIpCount']], on='bidder_id', how='left')
    t = pd.merge(test_data, b, on='bidder_id', how='left')
    mt = t.groupby('bidder_id').ipCount.mean().reset_index()
    mt = mt.rename(columns={'ipCount': 'meanIpCount'})
    st = t.groupby('bidder_id').ipCount.std().reset_index()
    st = st.rename(columns={'ipCount': 'stdIpCount'})
    tt = pd.merge(mt, st, on='bidder_id', how='left')
    # test_data = pd.merge(test_data, tt[['bidder_id', 'meanIpCount', 'stdIpCount']], on='bidder_id', how='left')

    temp = bid_data
    temp['count'] = 1
    deviceData = temp.groupby(['bidder_id', 'device'])['count'].sum().reset_index(name='count')
    deviceData = deviceData.pivot(index='bidder_id', columns='device', values='count')
    bidderList = deviceData.index.values

    for bidder in bidderList:
        if (deviceData.loc[bidder].mean() > 0):
            meanDeviceCount[bidder] = deviceData.loc[bidder].mean()

        if (deviceData.loc[bidder].std() > 0):
            stdDeviceCount[bidder] = deviceData.loc[bidder].std()

    train_data['meanBidsPerCountry'] = train_data.apply(lambda x: mergingFeature(x, meanCtryCount, bidderList), axis=1)
    test_data['meanBidsPerCountry'] = test_data.apply(lambda x: mergingFeature(x, meanCtryCount, bidderList), axis=1)

    train_data['stdBidsPerCountry'] = train_data.apply(lambda x: mergingFeature(x, stdCtryCount, bidderList), axis=1)
    test_data['stdBidsPerCountry'] = test_data.apply(lambda x: mergingFeature(x, stdCtryCount, bidderList), axis=1)

    train_data['meanBidsPerUrl'] = train_data.apply(lambda x: mergingFeature(x, meanUrlCount, bidderList), axis=1)
    test_data['meanBidsPerUrl'] = test_data.apply(lambda x: mergingFeature(x, meanUrlCount, bidderList), axis=1)

    train_data['stdBidsPerUrl'] = train_data.apply(lambda x: mergingFeature(x, stdUrlCount, bidderList), axis=1)
    test_data['stdBidsPerUrl'] = test_data.apply(lambda x: mergingFeature(x, stdUrlCount, bidderList), axis=1)

    train_data['meanBidsPerIp'] = train_data.apply(lambda x: mergingFeature(x, meanIpCount, bidderList), axis=1)
    test_data['meanBidsPerIp'] = test_data.apply(lambda x: mergingFeature(x, meanIpCount, bidderList), axis=1)

    train_data['stdBidsPerIp'] = train_data.apply(lambda x: mergingFeature(x, stdIpCount, bidderList), axis=1)
    test_data['stdBidsPerIp'] = test_data.apply(lambda x: mergingFeature(x, stdIpCount, bidderList), axis=1)

    train_data['meanBidsPerDevice'] = train_data.apply(lambda x: mergingFeature(x, meanDeviceCount, bidderList), axis=1)
    test_data['meanBidsPerDevice'] = test_data.apply(lambda x: mergingFeature(x, meanDeviceCount, bidderList), axis=1)

    train_data['stdBidsPerDevice'] = train_data.apply(lambda x: mergingFeature(x, stdDeviceCount, bidderList), axis=1)
    test_data['stdBidsPerDevice'] = test_data.apply(lambda x: mergingFeature(x, stdDeviceCount, bidderList), axis=1)
    return train_data, test_data

