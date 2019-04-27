from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt;  plt.rcdefaults()
from DataProcess import Data
import pickle
from sklearn.cross_validation import StratifiedKFold
from scipy import interp
import numpy as np
import pandas as pd
d = Data('data')
bid_data = d.bidData
train_data = d.trainData

def percentageBids():
    train_bids = pd.merge(bid_data, train_data,how='left',on='bidder_id')
    statData = train_bids.groupby('outcome')['bid_id'].count()
    '0 =  Human 1 = bot'
    per_human = statData[0] *100/(statData[0]+statData[1])
    per_bot = statData[1] *100/(statData[0]+statData[1])
    objects = ('% Human bids', '% Bot bids')
    y_pos = np.arange(len(objects))
    percent = [per_human, per_bot]
    plt.bar(y_pos, percent, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Percentage of Bids')
    plt.show()

def allGraphs():
    # train_bids = pd.merge(bid_data, train_data,how='left',on='bidder_id')
    # statData = train_bids.groupby('outcome')['bid_id'].count()
    # '0 =  Human 1 = bot'
    # per_human = statData[0] *100/(statData[0]+statData[1])
    # per_bot = statData[1] *100/(statData[0]+statData[1])
    # objects = ('% Human bids', '% Bot bids')
    # y_pos = np.arange(len(objects))
    # percent = [per_human, per_bot]
    # plt.bar(y_pos, percent, align='center', alpha=0.5)
    # plt.xticks(y_pos, objects)
    # plt.ylabel('Percentage of Bids')
    # plt.show()
    d = Data('data')
    bid_data = d.bidData
    bid_data['time'] = bid_data['time'] /100000000000
    bid_data.plot.scatter(x='time', y= 'bid_id')
    plt.scatter(bid_data['time'], bid_data['bid_id'], marker='o', color='r', alpha=1, s=400)
    plt.show()

def nnLossCurve(history):
    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
# allGraphs()

def roc_auc(train_features, classifier):
    X_train = train_features.drop(["bidder_id", "outcome"], axis=1)
    Y_train = train_features["outcome"]
    X = np.array(X_train)
    y = np.array(Y_train)
    cv = StratifiedKFold(y, n_folds=5)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    for i, (tran, tet) in enumerate(cv):
        probas_ = classifier.fit(X[tran], y[tran]).predict_proba(X[tet])
        fpr, tpr, thresholds = roc_curve(y[tet], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return mean_auc

def testGraph():
    x = ['a','b','c']
    y = [10,20,30]

    fig = plt.figure()

    width = .35
    ind = np.arange(len(y))
    plt.bar(ind, y, width=width)
    plt.xticks(ind, x)

    plt.show()

def mygraph():
    bid_data = pd.read_csv('data/bids.csv')
    feature_data = pd.read_csv('data/features.csv')
    all = pd.merge(bid_data, feature_data, how='left', on='bidder_id')

    bots = all.loc[all['outcome'] == 1]
    humans = all.loc[all['outcome'] == 0]

    merchandise = all['merchandise'].unique()
    count_bots = []
    count_humans = []
    for mer in merchandise:
        count_bots.append(bots[bots['merchandise'] == mer].count().unique()[0])
        count_humans.append(humans[humans['merchandise'] == mer].count().unique()[0])

    x = merchandise
    width = .35
    fig = plt.figure()
    ind = np.arange(len(x))
    g1 = plt.bar(ind, count_bots, width=width)
    g2 = plt.bar(ind + width, count_humans, width=width)
    plt.xticks(ind + width/2, x)
    fig.autofmt_xdate()

    plt.legend((g1[0], g2[0]), ('Bot', 'Human'))
    plt.ylabel('# of bids')
    plt.title('Total number of bids per category')
    plt.show()

def timegraph():
    bid_data = pd.read_csv('data/bids.csv')
    feature_data = pd.read_csv('data/train.csv')
    all = pd.merge(bid_data, feature_data, how='left', on='bidder_id')
    all = all.sort_values(by=['time'])

    bots = all.loc[all['outcome'] == 1]
    humans = all.loc[all['outcome'] == 0]
    time = all['time'].unique()
    count_bots = []
    count_humans = []
    count_all = []
    for t in time:
        count_bots.append(bots[bots['time'] == t].count().unique()[0])
        count_humans.append(humans[humans['time'] == t].count().unique()[0])
        count_all.append(all[all['time'] == t].count().unique()[0])
    g1 = plt.plot(time, count_all)
    g2 = plt.plot(time, count_bots)
    g3 = plt.plot(time, count_humans)
    plt.legend((g1[0], g2[0], g3[0]), ('All', 'Bot', 'Human'))
    plt.ylabel('# of bids')
    plt.xlabel('Time')
    plt.title('Total number of bids over one time interval')
    plt.show()

# timegraph()
