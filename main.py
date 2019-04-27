import sys
import model

def create_model():
    model.create_and_save()

def predict():
    print('Choose an algorithm: ')
    algo = input('1:LR\n2:Random Forest\n3:ada_boost\n4:gradient_boost\n5:bagged_tree\n6:extra_tree\n7:mlp\n8:rnn\n9:ensemble\n-->')
    print (algo)
    model.predict_score(algo)

if __name__ == '__main__':
    if (len(sys.argv) == 2):
        if(sys.argv[1] == 'create'):
            create_model()
        elif(sys.argv[1] == 'test'):
            predict()
    else:
        print ('Usage:\tmain.py <Create Model["create"] / Test on full test set["test"]>')
        sys.exit(0)