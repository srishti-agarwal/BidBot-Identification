

def numberofActions(line, dataGrouped, dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return dataGrouped[line['bidder_id']]

def mergingFeature(line, features, dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return features[line['bidder_id']]


def basicCountsPerUser(data):
    bid_data = data.bidData
    train_data = data.trainData
    test_data = data.testData

    bidderList = bid_data['bidder_id'].unique()
    countryCount = bid_data['country'].groupby(bid_data['bidder_id']).count()
    ipCount = bid_data['ip'].groupby(bid_data['bidder_id']).count()
    urlCount = bid_data['url'].groupby(bid_data['bidder_id']).count()
    deviceCount = bid_data['device'].groupby(bid_data['bidder_id']).count()
    auctionCount = bid_data['auction'].groupby(bid_data['bidder_id']).count()
    grBidCount = bid_data['bid_id'].groupby(bid_data['bidder_id']).count()
    grMerchandiseCount = bid_data['merchandise'].groupby(bid_data['bidder_id']).count()
    payAccCount_train = train_data['payment_account'].groupby(train_data['bidder_id']).count()
    payAccCount_test = test_data['payment_account'].groupby(test_data['bidder_id']).count()
    addressCount_train = train_data['address'].groupby(train_data['bidder_id']).count()
    addressCount_test = test_data['payment_account'].groupby(test_data['bidder_id']).count()

    train_data['nb0fCountry'] = train_data.apply(lambda x: numberofActions(x, countryCount, bidderList), axis=1)
    test_data['nb0fCountry'] = test_data.apply(lambda x: numberofActions(x, countryCount, bidderList), axis=1)

    train_data['nb0fIP'] = train_data.apply(lambda x: numberofActions(x, ipCount, bidderList), axis=1)
    test_data['nb0fIP'] = test_data.apply(lambda x: numberofActions(x, ipCount, bidderList), axis=1)

    train_data['nb0fURL'] = train_data.apply(lambda x: numberofActions(x, urlCount, bidderList), axis=1)
    test_data['nb0fURL'] = test_data.apply(lambda x: numberofActions(x, urlCount, bidderList), axis=1)

    train_data['nb0fDevice'] = train_data.apply(lambda x: numberofActions(x, deviceCount, bidderList), axis=1)
    test_data['nb0fDevice'] = test_data.apply(lambda x: numberofActions(x, deviceCount, bidderList), axis=1)

    train_data['nb0fAuction'] = train_data.apply(lambda x: numberofActions(x, auctionCount, bidderList), axis=1)
    test_data['nb0fAuction'] = test_data.apply(lambda x: numberofActions(x, auctionCount, bidderList), axis=1)

    train_data['nb0fBids'] = train_data.apply(lambda x: numberofActions(x, grBidCount, bidderList), axis=1)
    test_data['nb0fBids'] = test_data.apply(lambda x: numberofActions(x, grBidCount, bidderList), axis=1)

    train_data['nb0fMerch'] = train_data.apply(lambda x: numberofActions(x, grMerchandiseCount, bidderList), axis=1)
    test_data['nb0fMerch'] = test_data.apply(lambda x: numberofActions(x, grMerchandiseCount, bidderList), axis=1)

    train_data['nb0fPayAcc'] = train_data.apply(lambda x: numberofActions(x, payAccCount_train, bidderList), axis=1)
    test_data['nb0fPayAcc'] = test_data.apply(lambda x: numberofActions(x, payAccCount_test, bidderList), axis=1)

    train_data['nb0fAdress'] = train_data.apply(lambda x: numberofActions(x, addressCount_train, bidderList), axis=1)
    test_data['nb0fAdress'] = test_data.apply(lambda x: numberofActions(x, addressCount_test, bidderList), axis=1)

    return train_data, test_data

def bidsOnSelf(data):
    bid_data = data.bidData
    train_data = data.trainData
    test_data = data.testData

    bidderList = bid_data['bidder_id'].unique()
    tmp_data = bid_data.sort_values(['auction', 'time']).groupby(bid_data['auction'])
    # print (tmp_data.head(5))
    countBidsOnSelf = {}
    for t in tmp_data:
        prev = ''
        count = 0
        for b in t[1]['bidder_id']:
            if(b == prev):
                if(b in countBidsOnSelf.keys()):
                    count = countBidsOnSelf[b]
                    count += 1
            countBidsOnSelf[b] = count
            count = 0
            prev = b
    # timeDiff = {}
    # for t in tmp_data:
    #     prev = ''
    #     diff = 0
    #     for b in t[1]:
    #         print (b)
    #         if(b == prev):
    #             if(b in timeDiff.keys()):
    #                 diff = t[1]['time'] - prev[1]['time']
    #         timeDiff[b[1]['bidder_id']] = diff
    #         diff = 0
    #         prev = b
    # print (timeDiff)

    train_data['nb0fBidsOnSelf'] = train_data.apply(lambda x: mergingFeature(x, countBidsOnSelf, bidderList), axis=1)
    test_data['nb0fBidsOnSelf'] = test_data.apply(lambda x: mergingFeature(x, countBidsOnSelf, bidderList), axis=1)

    return train_data, test_data

