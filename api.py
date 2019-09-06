from nilmtk.dataset import DataSet
from nilmtk.metergroup import MeterGroup
import pandas as pd
from disaggregate import *
from six import iteritems
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
import numpy as np
import matplotlib.pyplot as plt
import datetime
from IPython.display import clear_output

class API():

    """
    The API ia designed for rapid experimentation with NILM Algorithms. 

    """

    def __init__(self,params):

        """
        Initializes the API with default parameters
        """
        self.power = {}
        self.sample_period = 1
        self.appliances = []
        self.methods = {}
        self.chunk_size = None
        self.pre_trained = False
        self.metrics = []
        self.train_datasets_dict = {}
        self.test_datasets_dict = {}
        #self.artificial_aggregate = False
        self.train_submeters = []
        self.train_mains = pd.DataFrame()
        self.test_submeters = []
        self.test_mains = pd.DataFrame()
        self.gt_overall = {}
        self.pred_overall = {}
        self.classifiers=[]
        self.DROP_ALL_NANS = True
        self.mae = pd.DataFrame()
        self.rmse = pd.DataFrame()
        self.experiment(params)

    
    def initialise(self,params):

        """
        Instantiates the API with the specified Parameters
        """
        for elems in params['power']:
            self.power = params['power']
        self.sample_period = params['sample_rate']
        for elems in params['appliances']:
            self.appliances.append(elems)
        
        self.pre_trained = ['pre_trained']
        self.train_datasets_dict = params['train']['datasets']
        self.test_datasets_dict = params['test']['datasets']
        self.metrics = params['test']['metrics']
        self.methods = params['methods']
        #self.artificial_aggregate = params.get('artificial_aggregate',self.artificial_aggregate)
        self.chunk_size = params.get('chunk_size',self.chunk_size)

    def experiment(self,params):
        """
        Calls the Experiments with the specified parameters
        """
        self.params=params
        self.initialise(params)
        self.store_classifier_instances()
        d=self.train_datasets_dict

        for model_name, clf in self.classifiers:
            # If the model is a neural net, it has an attribute n_epochs, Ex: DAE, Seq2Point
            print ("Started training for ",clf.MODEL_NAME)
            if hasattr(clf,'n_epochs'):
                epochs = clf.n_epochs
            # If it doesn't have the attribute n_epochs, this is executed. Ex: Mean, Zero
            else:
                epochs = 1
            # If the model has the filename specified for loading the pretrained model, then we don't need to load training data
            if clf.load_model_path:
                print (clf.MODEL_NAME," is loading the pretrained model")
                continue

            for q in range(epochs):
                print (clf, clf.chunk_wise_training, self.chunk_size)
                if clf.chunk_wise_training and self.chunk_size:
                    # The classifier can call partial fit on different chunks and refine the model
                    print ("Chunk training for ",clf.MODEL_NAME)
                    self.train_chunk_wise(clf,d)                

                else:
                    print ("Joint training for ",clf.MODEL_NAME)
                    self.train_jointly(clf,d)
            

            print ("Finished training for ",clf.MODEL_NAME)
            clear_output()

        d=self.test_datasets_dict
        if self.chunk_size:
            print ("Chunk Wise Testing for all algorithms")
            # It means that, predictions can also be done on chunks
            self.test_chunk_wise(d)

        else:
            print ("Joint Testing for all algorithms")
            self.test_jointly(d)

        # if clf.chunk_wise_training:
        #   # The classifier can call partial fit on different chunks and refine the model
        #   self.test_chunk_wise()              

        # else:
        #   self.test_jointly()

            
        # if params['chunk_size']:
        #   # This is for training and Testing in Chunks
        #   self.load_datasets_chunks()
        # else:
        #   # This is to load all the data from all buildings and use it for training and testing. This might not be possible to execute on computers with low specs
        #   self.load_datasets()
        
    def train_chunk_wise(self,clf,d):

        """
        This function loads the data from buildings and datasets with the specified chunk size and trains on each of them. 

        After the training process is over, it tests on the specified testing set whilst loading it in chunks.

        """
        # First, we initialize all the models   

            
        for dataset in d:
            print("Loading data for ",dataset, " dataset")          
            for building in d[dataset]['buildings']:
                train=DataSet(d[dataset]['path'])
                print("Loading building ... ",building)
                train.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
                mains_iterator = train.buildings[building].elec.mains().load(chunksize = self.chunk_size, physical_quantity='power', ac_type = self.power['mains'], sample_period=self.sample_period)
                appliance_iterators = [train.buildings[building].elec[app_name].load(chunksize = self.chunk_size, physical_quantity='power', ac_type=self.power['appliance'], sample_period=self.sample_period) for app_name in self.appliances]
                print(train.buildings[building].elec.mains())
                for chunk_num,chunk in enumerate (train.buildings[building].elec.mains().load(chunksize = self.chunk_size, physical_quantity='power', ac_type = self.power['mains'], sample_period=self.sample_period)):
                    #Dummry loop for executing on outer level. Just for looping till end of a chunk
                    print("Starting enumeration..........")
                    train_df = next(mains_iterator)
                    appliance_readings = []
                    for i in appliance_iterators:
                        try:
                            appliance_df = next(i)
                        except StopIteration:
                            appliance_df = pd.DataFrame()
                        appliance_readings.append(appliance_df)

                    if self.DROP_ALL_NANS:
                        train_df, appliance_readings = self.dropna(train_df, appliance_readings)
                    
                    if self.artificial_aggregate:
                        print ("Creating an Artificial Aggregate")
                        train_df = pd.DataFrame(np.zeros(appliance_readings[0].shape),index = appliance_readings[0].index,columns=appliance_readings[0].columns)
                        for app_reading in appliance_readings:
                            train_df+=app_reading
                    train_appliances = []

                    for cnt,i in enumerate(appliance_readings):
                        train_appliances.append((self.appliances[cnt],[i]))

                    self.train_mains = [train_df]
                    self.train_submeters = train_appliances
                    clf.partial_fit(self.train_mains,self.train_submeters)
                

        print("...............Finished the Training Process ...................")

    def test_chunk_wise(self,d):

        print("...............Started  the Testing Process ...................")

        for dataset in d:
            print("Loading data for ",dataset, " dataset")
            for building in d[dataset]['buildings']:
                test=DataSet(d[dataset]['path'])
                test.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
                mains_iterator = test.buildings[building].elec.mains().load(chunksize = self.chunk_size, physical_quantity='power', ac_type = self.power['mains'], sample_period=self.sample_period)
                appliance_iterators = [test.buildings[building].elec[app_name].load(chunksize = self.chunk_size, physical_quantity='power', ac_type=self.power['appliance'], sample_period=self.sample_period) for app_name in self.appliances]
                for chunk_num,chunk in enumerate (test.buildings[building].elec.mains().load(chunksize = self.chunk_size, physical_quantity='power', ac_type = self.power['mains'], sample_period=self.sample_period)):
                    test_df = next(mains_iterator)
                    appliance_readings = []
                    for i in appliance_iterators:
                        try:
                            appliance_df = next(i)
                        except StopIteration:
                            appliance_df = pd.DataFrame()

                        appliance_readings.append(appliance_df)

                    if self.DROP_ALL_NANS:
                        test_df, appliance_readings = self.dropna(test_df, appliance_readings)

                    if self.artificial_aggregate:
                        print ("Creating an Artificial Aggregate")
                        test_df = pd.DataFrame(np.zeros(appliance_readings[0].shape),index = appliance_readings[0].index,columns=appliance_readings[0].columns)
                        for app_reading in appliance_readings:
                            test_df+=app_reading

                    test_appliances = []

                    for cnt,i in enumerate(appliance_readings):
                        test_appliances.append((self.appliances[cnt],[i]))

                    self.test_mains = [test_df]
                    self.test_submeters = test_appliances
                    print("Results for Dataset {dataset} Building {building} Chunk {chunk_num}".format(dataset=dataset,building=building,chunk_num=chunk_num))
                    self.call_predict(self.classifiers)


    def train_jointly(self,clf,d):

        # This function has a few issues, which should be addressed soon
        print("............... Loading Data for training ...................")
        # store the train_main readings for all buildings
        self.train_mains = pd.DataFrame()
        self.train_submeters = [pd.DataFrame() for i in range(len(self.appliances))]
        for dataset in d:
            print("Loading data for ",dataset, " dataset")
            train=DataSet(d[dataset]['path'])
            for building in d[dataset]['buildings']:
                print("Loading building ... ",building)
                train.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
                train_df = next(train.buildings[building].elec.mains().load(physical_quantity='power', ac_type=self.power['mains'], sample_period=self.sample_period))
                train_df = train_df[[list(train_df.columns)[0]]]
                appliance_readings = []
                
                for appliance_name in self.appliances:
                    appliance_df = next(train.buildings[building].elec[appliance_name].load(physical_quantity='power', ac_type=self.power['appliance'], sample_period=self.sample_period))
                    appliance_df = appliance_df[[list(appliance_df.columns)[0]]]
                    appliance_readings.append(appliance_df)

                if self.DROP_ALL_NANS:
                    train_df, appliance_readings = self.dropna(train_df, appliance_readings)
                print ("Train Jointly")
                print (train_df.shape, appliance_readings[0].shape, train_df.columns,appliance_readings[0].columns )
                self.train_mains=self.train_mains.append(train_df)
                for i,appliance_name in enumerate(self.appliances):
                    self.train_submeters[i] = self.train_submeters[i].append(appliance_readings[i])

        appliance_readings = []
        for i,appliance_name in enumerate(self.appliances):
            appliance_readings.append((appliance_name, [self.train_submeters[i]]))

        self.train_mains = [self.train_mains]
        self.train_submeters = appliance_readings   
        clf.partial_fit(self.train_mains,self.train_submeters)

    def test_jointly(self,d):


        # store the test_main readings for all buildings
        for dataset in d:
            print("Loading data for ",dataset, " dataset")
            test=DataSet(d[dataset]['path'])
            for building in d[dataset]['buildings']:
                test.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
                test_mains=next(test.buildings[building].elec.mains().load(physical_quantity='power', ac_type=self.power['mains'], sample_period=self.sample_period))
                appliance_readings=[]

                #print (self.appliances  , self.power['appliance'],self.sample_period)
                
                #elec = test.buildings[building].elec
                #df=next((elec.select_using_appliances(type='fridge').load(physical_quantity='power', ac_type=['apparent','active'], sample_period=60)))

                #print (df.shape)
                for appliance in self.appliances:
                    test_df=next((test.buildings[building].elec[appliance].load(physical_quantity='power', ac_type=self.power['appliance'], sample_period=self.sample_period)))
                    appliance_readings.append(test_df)

                
                if self.DROP_ALL_NANS:
                    test_mains, appliance_readings = self.dropna(test_mains, appliance_readings)

                self.test_mains = [test_mains]
                for i, appliance_name in enumerate(self.appliances):
                    self.test_submeters.append((appliance_name,[appliance_readings[i]]))
                
                self.call_predict(self.classifiers)
            

    def dropna(self,mains_df, appliance_dfs):
        """
        Drops the missing values in the Mains reading and appliance readings and returns consistent data by copmuting the intersection
        """
        print ("Dropping missing values")

        # The below steps are for making sure that data is consistent by doing intersection across appliances
        mains_df = mains_df.dropna()
        for i in range(len(appliance_dfs)):
            appliance_dfs[i] = appliance_dfs[i].dropna()
        ix = mains_df.index
        for  app_df in appliance_dfs:
            ix = ix.intersection(app_df.index)
        mains_df = mains_df.loc[ix]
        new_appliances_list = []
        for app_df in appliance_dfs:
            new_appliances_list.append(app_df.loc[ix])
        return mains_df,new_appliances_list
    
    
    def store_classifier_instances(self):

        """
        This function is reponsible for initializing the models with the specified model parameters
        """
        method_dict={}
        for i in self.methods:
            model_class = globals()[i]
            method_dict[i] = model_class(self.methods[i])

        # method_dict={'CO':CombinatorialOptimisation(self.method_dict['CO']),
        #             'FHMM':FHMM(self.method_dict['FHMM']),
        #             'DAE':DAE(self.method_dict['DAE']),
        #             'Mean':Mean(self.method_dict['Mean']),
        #             'Zero':Zero(self.method_dict['Zero']),
        #             'Seq2Seq':Seq2Seq(self.method_dict['Seq2Seq']),
        #             'Seq2Point':Seq2Point(self.method_dict['Seq2Point']),
        #             'DSC':DSC(self.method_dict['DSC']),
        #              'AFHMM':AFHMM(self.method_dict['AFHMM']),
        #              'AFHMM_SAC':AFHMM_SAC(self.method_dict['AFHMM_SAC']),              
        #              'RNN':RNN(self.method_dict['RNN'])
        #             }

        for name in self.methods:
            if 1:
                clf=method_dict[name]
                self.classifiers.append((name,clf))
            else:
                print ("\n\nThe method {model_name} specied does not exist. \n\n".format(model_name=i))
    
    def call_predict(self,classifiers):

        """
        This functions computers the predictions on the self.test_mains using all the trained models and then compares different learn't models using the metrics specified
        """
        
        pred_overall={}
        gt_overall={}
        for name,clf in classifiers:
            gt_overall,pred_overall[name]=self.predict(clf,self.test_mains,self.test_submeters, self.sample_period,'Europe/London')

        self.gt_overall=gt_overall
        self.pred_overall=pred_overall

        if gt_overall.size==0:
            print ("No samples found in ground truth")
            return None

        for i in gt_overall.columns:
            plt.figure()
            
            plt.plot(self.test_mains[0],label='Mains reading')
            plt.plot(gt_overall[i],label='Truth')
            for clf in pred_overall:
                
                plt.plot(pred_overall[clf][i],label=clf)
            plt.title(i)
            plt.legend()
            plt.show()



        for metric in self.metrics:
            
            if metric=='f1-score':
                f1_score={}
                
                for clf_name,clf in classifiers:
                    f1_score[clf_name] = self.compute_f1_score(gt_overall, pred_overall[clf_name])
                f1_score = pd.DataFrame(f1_score)
                print("............ " ,metric," ..............")
                print(f1_score) 
                
            elif metric=='rmse':
                rmse = {}
                for clf_name,clf in classifiers:
                    rmse[clf_name] = self.compute_rmse(gt_overall, pred_overall[clf_name])
                rmse = pd.DataFrame(rmse)
                self.rmse = rmse
                print("............ " ,metric," ..............")
                print(rmse) 

            elif metric=='mae':
                mae={}
                for clf_name,clf in classifiers:
                    mae[clf_name] = self.compute_mae(gt_overall, pred_overall[clf_name])
                mae = pd.DataFrame(mae)
                self.mae = mae
                print("............ " ,metric," ..............")
                print(mae)  

            elif metric == 'rel_error':
                rel_error={}
                for clf_name,clf in classifiers:
                    rel_error[clf_name] = self.compute_rel_error(gt_overall, pred_overall[clf_name])
                rel_error = pd.DataFrame(rel_error)
                print("............ " ,metric," ..............")
                print(rel_error)            
            else:
                print ("The requested metric {metric} does not exist.".format(metric=metric))
                    
    def predict(self, clf, test_elec, test_submeters, sample_period, timezone):
        
        """
        Generates predictions on the test dataset using the specified classifier.
        """
        
        # "ac_type" varies according to the dataset used. 
        # Make sure to use the correct ac_type before using the default parameters in this code.   
        
            
        pred_list = clf.disaggregate_chunk(test_elec)

        # It might not have time stamps sometimes due to neural nets
        # It has the readings for all the appliances

        concat_pred_df = pd.concat(pred_list,axis=0)

        gt = {}
        for meter,data in test_submeters:
                concatenated_df_app = pd.concat(data,axis=1)
                index = concatenated_df_app.index
                gt[meter] = pd.Series(concatenated_df_app.values.flatten(),index=index)

        gt_overall = pd.DataFrame(gt, dtype='float32')
        
        pred = {}

        for app_name in concat_pred_df.columns:

            app_series_values = concat_pred_df[app_name].values.flatten()

            # Neural nets do extra padding sometimes, to fit, so get rid of extra predictions

            app_series_values = app_series_values[:len(gt_overall[app_name])]

            #print (len(gt_overall[app_name]),len(app_series_values))

            pred[app_name] = pd.Series(app_series_values, index = gt_overall.index)

        pred_overall = pd.DataFrame(pred,dtype='float32')


        #gt[i] = pd.DataFrame({k:v.squeeze() for k,v in iteritems(gt[i]) if len(v)}, index=next(iter(gt[i].values())).index).dropna()

        # If everything can fit in memory

        #gt_overall = pd.concat(gt)
        # gt_overall.index = gt_overall.index.droplevel()
        # #pred_overall = pd.concat(pred)
        # pred_overall.index = pred_overall.index.droplevel()

        # Having the same order of columns
        # gt_overall = gt_overall[pred_overall.columns]

        # #Intersection of index
        # gt_index_utc = gt_overall.index.tz_convert("UTC")
        # pred_index_utc = pred_overall.index.tz_convert("UTC")
        # common_index_utc = gt_index_utc.intersection(pred_index_utc)

        # common_index_local = common_index_utc.tz_convert(timezone)
        # gt_overall = gt_overall.loc[common_index_local]
        # pred_overall = pred_overall.loc[common_index_local]
        # appliance_labels = [m for m in gt_overall.columns.values]
        # gt_overall.columns = appliance_labels
        # pred_overall.columns = appliance_labels
        return gt_overall, pred_overall


    # metrics
    def compute_mae(self,gt,pred):
        """
        Computes the Mean Absolute Error between Ground truth and Prediction
        """

        mae={}
        for appliance in gt.columns:
            mae[appliance]=mean_absolute_error(gt[appliance],pred[appliance])
        return pd.Series(mae)


    def compute_rmse(self,gt, pred):
        """
        Computes the Root Mean Squared Error between Ground truth and Prediction
        """
        rms_error = {}
        for appliance in gt.columns:
            rms_error[appliance] = np.sqrt(mean_squared_error(gt[appliance], pred[appliance]))
        #print (gt['sockets'])
        #print (pred[])
        return pd.Series(rms_error)
    
    def compute_f1_score(self,gt, pred):
        """
        Computes the F1 Score between Ground truth and Prediction
        """ 
        f1 = {}
        gttemp={}
        predtemp={}
        for appliance in gt.columns:
            gttemp[appliance] = np.array(gt[appliance])
            gttemp[appliance] = np.where(gttemp[appliance]<10,0,1)
            predtemp[appliance] = np.array(pred[appliance])
            predtemp[appliance] = np.where(predtemp[appliance]<10,0,1)
            f1[appliance] = f1_score(gttemp[appliance], predtemp[appliance])
        return pd.Series(f1)

    def compute_rel_error(self,gt,pred):

        """
        Computes the Relative Error between Ground truth and Prediction
        """
        # THe metric is wrong
        rel_error={}
        for appliance in gt.columns:
            rel_error[appliance] = np.mean(np.abs((gt[appliance] - pred[appliance])/(gt[appliance] + 1))) * 100
        # The extra 1 is added for the case where gt is zero
        return pd.Series(rel_error) 
