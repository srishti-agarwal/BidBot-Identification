from features import User, User2, Auction, Miscellaneous
from features.Time import timeStartEndDiff, bidsPerTime
from DataProcess import Data
from graphs import roc_auc, nnLossCurve
from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn.ensemble import VotingClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_features():
    print ("Loading train features")
    train_feature_pkl = open('model/train_features.pkl', 'rb')
    train_features = pickle.load(train_feature_pkl)
    print ("Loaded train features")

    print ("Loading test features")
    test_feature_pkl = open('model/test_features.pkl', 'rb')
    test_features = pickle.load(test_feature_pkl)
    print ("Loaded test features")

    return train_features, test_features

def logistic_regr():
    train_features, test_features = load_features()
    train_features = train_features.fillna(value=0)
    test_features  = test_features.fillna(value=0)
    X_train = train_features.drop(["bidder_id", "outcome"], axis=1)
    Y_train = train_features["outcome"]
    X_test = test_features.drop(["bidder_id"], axis=1)
    print ("Training logistic regression model")
    logisticRegr = LogisticRegression()
    print ("Model trained")
    print ("Cross validation score (Logistic Regression : ")
    cv_score = np.nanmean(cross_val_score(logisticRegr, X_train, Y_train, cv=5, scoring='roc_auc'))
    print (cv_score)

    print ("Generating submission file")
    logisticRegr.fit_transform(X_train, Y_train)
    prediction = logisticRegr.predict_proba(X_test)
    test_features['prediction'] = prediction[:, 1]
    test_features[['bidder_id', 'prediction']].to_csv('data/submission_lr.csv', index=False)
    print ("Output file successfully created")

    print ("Generating auc curve and auc score")
    auc = roc_auc(train_features, logisticRegr)
    print ("AUC score : "+str(auc))

def random_forest():
    train_features, test_features = load_features()
    train_features = train_features.fillna(value=0)
    test_features = test_features.fillna(value=0)
    X_train = train_features.drop(["bidder_id", "outcome"], axis=1)
    Y_train = train_features["outcome"]
    X_test = test_features.drop(["bidder_id"], axis=1)
    print ("Training random forest model")
    randomForest = RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=1)
    print ("Model trained")
    print ("Cross validation score (Random Forest) : ")
    cv_score = np.mean(cross_val_score(randomForest, X_train, Y_train, cv=5, scoring='roc_auc'))
    print (cv_score)

    print ("Generating submission file")
    randomForest.fit(X_train, Y_train)
    prediction = randomForest.predict_proba(X_test)
    test_features['prediction'] = prediction[:, 1]
    test_features[['bidder_id', 'prediction']].to_csv('data/submission_rf.csv', index=False)
    print ("Output file successfully created")

    print ("Generating auc curve and auc score")
    auc = roc_auc(train_features, randomForest)
    print ("AUC score : " + str(auc))
    x = X_train.columns
    y = randomForest.feature_importances_
    ind = np.arange(len(y))
    fig = plt.figure()
    plt.bar(ind, y, width=0.35)
    plt.xticks(ind, x)
    plt.ylabel("Importance")
    plt.title("Feature Importance")
    fig.autofmt_xdate()
    plt.show()

def bagged_tree():
    train_features, test_features = load_features()
    train_features = train_features.fillna(value=0)
    test_features = test_features.fillna(value=0)
    X_train = train_features.drop(["bidder_id", "outcome"], axis=1)
    Y_train = train_features["outcome"]
    X_test = test_features.drop(["bidder_id"], axis=1)
    print ("Training Bagged forest classifier model")
    cart = DecisionTreeClassifier()
    bag_class = BaggingClassifier(base_estimator=cart, n_estimators=3000)
    print ("Model trained")
    print ("Cross validation score (Bagged Forest) : ")
    cv_score = np.mean(cross_val_score(bag_class, X_train, Y_train, cv=5, scoring='roc_auc'))
    print (cv_score)

    print ("Generating submission file")
    bag_class.fit(X_train, Y_train)
    prediction = bag_class.predict_proba(X_test)
    test_features['prediction'] = prediction[:, 1]
    test_features[['bidder_id', 'prediction']].to_csv('data/submission_bagged.csv', index=False)
    print ("Output file successfully created")

    print ("Generating auc curve and auc score")
    auc = roc_auc(train_features, bag_class)
    print ("AUC score : " + str(auc))

def ada_boost():
    train_features, test_features = load_features()
    train_features = train_features.fillna(value=0)
    test_features = test_features.fillna(value=0)
    X_train = train_features.drop(["bidder_id", "outcome"], axis=1)
    Y_train = train_features["outcome"]
    X_test = test_features.drop(["bidder_id"], axis=1)
    print ("Training AdaBoost forest model")
    adaBoost = AdaBoostClassifier(n_estimators=3000, learning_rate=0.001)
    print ("Model trained")
    print ("Cross validation score (AdaBoost) : ")
    cv_score = np.mean(cross_val_score(adaBoost, X_train, Y_train, cv=5, scoring='roc_auc'))
    print (cv_score)

    print ("Generating submission file")
    adaBoost.fit(X_train, Y_train)
    prediction = adaBoost.predict_proba(X_test)
    test_features['prediction'] = prediction[:, 1]
    test_features[['bidder_id', 'prediction']].to_csv('data/submission_ada.csv', index=False)
    print ("Output file successfully created")

    print ("Generating auc curve and auc score")
    auc = roc_auc(train_features, adaBoost)
    print ("AUC score : " + str(auc))

def gradient_boost():
    train_features, test_features = load_features()
    train_features = train_features.fillna(value=0)
    test_features = test_features.fillna(value=0)
    X_train = train_features.drop(["bidder_id", "outcome"], axis=1)
    Y_train = train_features["outcome"]
    X_test = test_features.drop(["bidder_id"], axis=1)
    print ("Training Gradient forest model")
    graBoost = GradientBoostingClassifier(n_estimators=500, learning_rate=0.001,max_depth=6, min_samples_leaf=1, max_features='sqrt')
    print ("Model trained")
    print ("Cross validation score (GradientBoost) : ")
    cv_score = np.mean(cross_val_score(graBoost, X_train, Y_train, cv=5, scoring='roc_auc'))
    print (cv_score)

    print ("Generating submission file")
    graBoost.fit(X_train, Y_train)
    print (graBoost.feature_importances_)
    prediction = graBoost.predict_proba(X_test)
    test_features['prediction'] = prediction[:, 1]
    test_features[['bidder_id', 'prediction']].to_csv('data/submission_gradient.csv', index=False)
    print ("Output file successfully created")

    print ("Generating auc curve and auc score")
    auc = roc_auc(train_features, graBoost)
    print ("AUC score : " + str(auc))

def extra_tree():
    train_features, test_features = load_features()
    train_features = train_features.fillna(value=0)
    test_features = test_features.fillna(value=0)
    X_train = train_features.drop(["bidder_id", "outcome"], axis=1)
    Y_train = train_features["outcome"]
    X_test = test_features.drop(["bidder_id"], axis=1)
    print("Training extra_tree model")
    extraTree = ExtraTreesClassifier(n_estimators=3000, max_features=10)
    print("Model trained")
    print("Cross validation score (extra_tree) : ")
    cv_score = np.mean(cross_val_score(extraTree, X_train, Y_train, cv=5, scoring='roc_auc'))
    print(cv_score)

    print("Generating submission file")
    extraTree.fit(X_train, Y_train)
    prediction = extraTree.predict_proba(X_test)
    test_features['prediction'] = prediction[:, 1]
    test_features[['bidder_id', 'prediction']].to_csv('data/submission_extra_tree.csv', index=False)
    print("Output file successfully created")

    print("Generating auc curve and auc score")
    auc = roc_auc(train_features, extraTree)
    print("AUC score : " + str(auc))


def mlp():
    train_features, test_features = load_features()
    train_features = train_features.fillna(value=0)
    train_data, test_data = train_test_split(train_features, test_size=0.30, random_state=False)
    X_train = train_data.drop(['outcome', 'bidder_id'], axis=1)
    Y_train = train_data['outcome']
    X_test = test_data.drop(['outcome', 'bidder_id'], axis=1)
    Y_test = test_data['outcome']
    clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='adaptive',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
    clf.fit(X_train, Y_train)
    pred = clf.predict_proba(X_test)
    print ("predictions: ",pred)
    fpr, tpr, thresholds = roc_curve(Y_test, pred[:,1])
    roc_auc = auc(fpr, tpr)
    print ("roc_auc: ", roc_auc)

def rnn():
    train_features, test_features = load_features()
    train_features = train_features.fillna(value=0)
    train_data, test_data = train_test_split(train_features, test_size=0.20, random_state=False)
    X_train = train_data.drop(['outcome', 'bidder_id'], axis=1)
    Y_train = train_data['outcome']
    X_test = test_data.drop(['outcome', 'bidder_id'], axis=1)
    Y_test = test_data['outcome']
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    Y_train = scaler.fit_transform(Y_train)
    seed = 7
    np.random.seed(seed)
    model = Sequential()
    model.add(Dense(10, input_dim = 40, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1,init='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    print('\n')
    # Fit the model
    hist = model.fit(X_train, Y_train, epochs=15, validation_split=0.10)
    nnLossCurve(hist)
    # evaluate the model
    pred = model.predict_proba(X_test)
    print("predictions: ", pred)
    fpr, tpr, thresholds = roc_curve(Y_test, pred)
    roc_auc = auc(fpr, tpr)
    print("roc_auc: ", roc_auc)

def create_and_save():
    print ("Loading data...")
    data = Data('data')
    print ("Extracting features...")
    print ("1. Extracting basic counts per user")
    data.trainData, data.testData = User.basicCountsPerUser(data)
    print ("2. Extracting basic unique counts per user")
    data.trainData, data.testData = User2.basicUniqueCountsPerUser(data)
    print ("3. Extracting granular merchandise")
    data.trainData, data.testData = User2.granularMerchandise(data)
    print ("4. Extracting bids on self")
    data.trainData, data.testData = User.bidsOnSelf(data)
    print ("5. Extracting auction features")
    data.trainData, data.testData = Auction.findAuctionFeatures(data)
    print ("6. Extracting miscellaneous features")
    data.trainData, data.testData = Miscellaneous.findMiscellaneousFeatures(data)
    print("7. Extracting temporal features")
    data.trainData, data.testData = timeStartEndDiff(data)
    data.trainData, data.testData = bidsPerTime(data)
    print ("Saving train features")
    print (data.testData.shape)
    data.trainData.to_csv('trained_features.csv', index=False)
    train_features = data.trainData.drop(["payment_account", "address"], axis=1)
    feature_pkl_filename = 'model/train_features.pkl'
    feature_pkl = open(feature_pkl_filename, 'wb')
    pickle.dump(train_features, feature_pkl)
    feature_pkl.close()
    print ("Train Features saved")
    print ("Saving test features")
    test_features = data.testData.drop(["payment_account", "address"], axis=1)
    feature_pkl_filename = 'model/test_features.pkl'
    feature_pkl = open(feature_pkl_filename, 'wb')
    pickle.dump(test_features, feature_pkl)
    feature_pkl.close()
    print ("Test Features saved")

def ensemble():
    train_features, test_features = load_features()
    train_features = train_features.fillna(value=0)
    test_features = test_features.fillna(value=0)
    X_train = train_features.drop(["bidder_id", "outcome"], axis=1)
    Y_train = train_features["outcome"]
    X_test = test_features.drop(["bidder_id"], axis=1)
    print ("Training ensemble model")
    randomForest = RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=1)
    graBoost = GradientBoostingClassifier(n_estimators=500, learning_rate=0.001,max_depth=6, min_samples_leaf=1, max_features='sqrt')
    extraTree = ExtraTreesClassifier(n_estimators=500, max_features=10)
    ensemble = VotingClassifier(estimators=[('rf',randomForest),('gb', graBoost),('et',extraTree)],voting='soft',weights=[2.5,1,0.5])

    print ("Cross validation score (Ensemble) : ")
    cv_score = np.mean(cross_val_score(ensemble, X_train, Y_train, cv=5, scoring='roc_auc'))
    print (cv_score)

    print ("Generating submission file")
    print ("Model trained")

    ensemble.fit(X_train, Y_train)
    prediction = ensemble.predict_proba(X_test)
    test_features['prediction'] = prediction[:, 1]
    test_features[['bidder_id', 'prediction']].to_csv('data/submission_re.csv', index=False)
    print ("Output file successfully created")

    print ("Generating auc curve and auc score")
    auc = roc_auc(train_features, ensemble)
    print ("AUC score : " + str(auc))


def predict_score(algo):
    algo = int(algo)
    options = {
        1 : logistic_regr,
        2 : random_forest,
        3 : ada_boost,
        4 : gradient_boost,
        5 : bagged_tree,
        6 : extra_tree,
        7 : mlp,
        8 : rnn,
        9 : ensemble,
    }
    options[algo]()
