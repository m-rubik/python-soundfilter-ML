import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

class AI():

    def __init__(self):
        self.name= "MLP"
        self.df_talking = None
        self.df_noise = None

    def load_data_talking(self, filename):
        # p = Path(filename).with_suffix('.csv')
        if self.df_talking == None:
            self.df_talking = pd.read_csv(filename+".csv")
        else:
            df = pd.read_csv(filename+".csv")
            self.df_talking= pd.concat([self.df_talking, df], ignore_index=True)

    def load_data_noise(self, filename):
        # p = Path(filename).with_suffix('.csv')
        if self.df_noise == None:
            self.df_noise = pd.read_csv(filename+".csv")
        else:
            df = pd.read_csv(filename+".csv")
            self.df_noise= pd.concat([self.df_noise, df], ignore_index=True)

    def merge_datasets(self):
        self.df_talking['is_talking'] = 1
        self.df_noise['is_talking'] = 0
       
        self.df = pd.concat([self.df_talking, self.df_noise], ignore_index=True)

        self.df.to_csv("df_merged.csv")

    def generate_features(self, test_fraction=0.2):
        d = np.isfinite(self.df)
        d.to_csv("df_merged_inf.csv")
        y = self.df['is_talking']
        X = self.df.drop(columns=['is_talking'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_fraction)

    def test_model(self, save_threshold=80):
        """
        Method for testing the performance of
        any given model.
        """

        print("Detailed classification report:")
        self.y_true, self.y_pred = self.y_test, self.clf.predict(self.X_test)

        print("Detailed classification report:")
        print(classification_report(self.y_true, self.y_pred))
        
        self.plot_confusion_matrix()

        self.confidence = metrics.r2_score(self.y_test, self.y_pred)
        print("R^2:", str(round(self.confidence*100, 2)) +
                "%. (Ideally as close to 0 as possible)")
        # NOTE: THESE CAN ONLY BE INTEGERS. YOU CANNOT TEST AGAINST FLOATS
        self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        model_accuracy = round(self.accuracy*100, 2)
        print("Accuracy:", str(model_accuracy) +
                "%. (Ideally as close to 100 as possible)")
        self.confusion_matrix = metrics.confusion_matrix(self.y_test, self.y_pred)

        # testIndex = self.df.shape[0]-len(self.y_pred)
        # test_df = self.df.reset_index()
        # test_df = test_df[testIndex:]
        # test_df['prediction'] = self.y_pred

        # print(sum(1 for val in y_pred if val == 1))
        # print(sum(1 for val in y_pred if val == -1))
        # print(len(y_pred))

        if model_accuracy >= save_threshold:
            self.name = self.name+"_"+str(int(round(self.accuracy*100, 0)))
            print("Saving model as", self.name)
            # export_model(self)
            model_file = self.name+".pickle"
            self.save_model(model_file, self.clf)

    def plot_confusion_matrix(self):
        cf_matrix = metrics.confusion_matrix(self.y_true, self.y_pred)
        ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
        ax.set_title('Confusion Matrix\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');
        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(['False','True'])
        ax.yaxis.set_ticklabels(['False','True'])
        ## Display the visualization of the Confusion Matrix.
        plt.show()

    def generate_voting(self):
        """
        Method to generate a voting
        classifer model.
        """

        svc = LinearSVC(max_iter=10000)
        rfor = RandomForestClassifier(n_estimators=100)
        knn = KNeighborsClassifier()
        self.clf = VotingClassifier(
            [('lsvc', svc), ('knn', knn), ('rfor', rfor)])
        print("Training classifier...")
        for classifier, label in zip([svc, knn, rfor], ['lsvc', 'knn', 'rfor']):
            scores = cross_val_score(
                classifier, self.X_train, self.y_train, cv=5, scoring='accuracy')
            print("Accuracy: %0.2f (+/- %0.2f) [%s]" %
                    (scores.mean(), scores.std(), label))
        self.clf.fit(self.X_train, self.y_train)
        self.test_model()

    def generate_mlp(self):
        """
        Method for generating a Multilayer perceptron
        Neural Network using Sklearn's MLP.
        """

        mlp = MLPClassifier(max_iter=1000, verbose=False)
        # print(self.X_train.shape)
        self.num_input_neurons = (self.X_train.shape)[1]
        self.num_output_neurons = 2  # Talking or no talking
        self.num_hidden_nodes = round(
            self.num_input_neurons*(2/3) + self.num_output_neurons)
        self.num_hn_perlayer = round(self.num_hidden_nodes/3)

        # Hyper-parameter optimization
        parameter_space = {
            'hidden_layer_sizes': [(self.num_hn_perlayer, self.num_hn_perlayer, self.num_hn_perlayer),
                                    (self.num_hidden_nodes,),
                                    (self.num_hn_perlayer, self.num_hn_perlayer, self.num_hn_perlayer, self.num_hn_perlayer)],
            'activation': ['tanh', 'relu', 'logistic'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
        }
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        print("Performing a gridsearch to find the optimal NN hyper-parameters.")
        self.clf = GridSearchCV(
            mlp, parameter_space, n_jobs=-1, cv=3, verbose=True)
        self.clf.fit(self.X_train, self.y_train)

        # Print results to console
        print('Best parameters found:\n', self.clf.best_params_)
        # print("Grid scores on development set:")
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
        #     mlp.hidden_layer_sizes = (
        #         self.num_hn_perlayer, self.num_hn_perlayer, self.num_hn_perlayer)
        #     self.clf = mlp.fit(self.X_train, self.y_train)
        self.test_model()

    def generate_bagging(self):
        """
        Method to generate a bagging
        classifer model.
        """

        self.clf = AdaBoostClassifier(
            DecisionTreeClassifier(), n_estimators=30)
        self.clf.fit(self.X_train, self.y_train)
        print(self.clf.score(self.X_test, self.y_test))
        self.test_model()

    def save_model(self, filename, data):
        """!
        Pickle a python object into a serialized (pickle) object.
        @param filename: The name of the pickle object to generate/overwrite
        @param data: Python object to get pickled
        """

        try:
            with open(filename, 'wb+') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(e)
            return 1
        return 0

    def import_model(self, filename):
        """!
        Import a pickled object into python object.
        @param filename: Path of the pickled object
        """

        if Path(filename).is_file():
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.clf = data
        else:
            self.clf = None

    def generate_keras_model(self):
        pass

    def keras_model(self):
        import tensorflow as tf
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.SimpleRNN(128, input_shape = (1,1)))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=4, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
        model.summary()

if __name__ == "__main__":
    a = AI()
    a.load_data_talking("sample_talking")
    a.load_data_noise("sample_typing")
    a.merge_datasets()

    ## To make a new model
    a.generate_features(0.2)
    a.generate_voting()
    # a.generate_mlp()

    # ##  To test an existing model
    # a.generate_features(0.8)
    # a.import_model("MLP_99.pickle")
    # a.test_model()

