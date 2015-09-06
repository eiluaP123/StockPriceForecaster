import numpy as np
import datetime as dt
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da
import sys, csv, math
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

def getFeatures(traindata):
	features_sel = []
	Y_col = []
	length = len(traindata)-100-5
	#print "length",length
	for file in range(np.size(traindata,1)):
		for date in range(100,np.size(traindata,0)-5):
			lookback = traindata[date-100:date,file]
			index = date-100
			std = np.std(lookback)
			mean = np.mean(lookback)
			amp = np.amax(lookback) - np.amin(lookback)
			number_above_mean = 0
			for i in range(np.size(lookback,0)):
				if lookback[i] > mean:
					number_above_mean += 1
			phase = traindata[date,file] / amp
			one_day_price_change = 0
			count = 0
			for j in range(index+1,date):
				one_day_price_change += traindata[date,file] - traindata[date-1,file]
				count += 1
			one_day_price_change= one_day_price_change/count
			
			features_sel.append([std, amp, number_above_mean, one_day_price_change])
			Y_col.append([traindata[date+5,file] - traindata[date,file]])
			
	return np.array(features_sel), np.array(Y_col)
	
def main():
	# To create the list of symbols for training data and test data
	'''sys.stdout = open("trainingdata.txt","w")
	sys.stdout2 = open("testdata.txt", "w")
	for i in range(0,200):
		if i<10:
			filename = "ML4T-00"+str(i)
		elif i>9 and i<100:
			filename = "ML4T-0"+str(i)
		else:
			filename = "ML4T-"+str(i)
		sys.stdout.write(filename)
		sys.stdout.write("\n")
	sys.stdout.close()
	sys.stdout2.write("ML4T-292")
	sys.stdout2.write("\n")
	sys.stdout2.write("ML4T-329")
	sys.stdout2.close()'''
	
	dataobj = da.DataAccess('Yahoo')
	ls_symbols_train = dataobj.get_symbols_from_list("trainingdata")
	ls_symbols_test = dataobj.get_symbols_from_list("testdata")
	ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
	dt_start_train = dt.datetime(2001, 1, 1)
	dt_end_train = dt.datetime(2005, 12, 31)
	ldt_timestamps_train = du.getNYSEdays(dt_start_train, dt_end_train, dt.timedelta(hours=16))
	ldf_data_train = dataobj.get_data(ldt_timestamps_train, ls_symbols_train, ls_keys)
	d_data_train = dict(zip(ls_keys, ldf_data_train))
	'''for s_key in ls_keys:
		d_data_train[s_key] = d_data_train[s_key].fillna(method='ffill')
		d_data_train[s_key] = d_data_train[s_key].fillna(method='bfill')
		d_data_train[s_key] = d_data_train[s_key].fillna(1.0)'''
	close_train = np.array(d_data_train['close'])
	#close_train = tsu.returnize0(d_data_train['close'])
	#close_train = np.array(close_train)
	
	dt_start_test = dt.datetime(2006, 1, 1)
	dt_end_test = dt.datetime(2007, 12, 31)
	ldt_timestamps_test = du.getNYSEdays(dt_start_test, dt_end_test, dt.timedelta(hours=16))
	ldf_data_test = dataobj.get_data(ldt_timestamps_test, ls_symbols_test, ls_keys)
	d_data_test = dict(zip(ls_keys, ldf_data_test))
	'''for s_key in ls_keys:
		d_data_test[s_key] = d_data_test[s_key].fillna(method='ffill')
		d_data_test[s_key] = d_data_test[s_key].fillna(method='bfill')
		d_data_test[s_key] = d_data_test[s_key].fillna(1.0)'''
	close_test = np.array(d_data_test['close'])
	#close_test = tsu.returnize0(d_data_test['close'])
	#close_test = np.array(close_test)
	
	Xtrain, Ytrain = getFeatures(close_train)
	#print close_test[:,0].shape
	#print close_test[:,1].shape
	test292 = np.reshape(close_test[:,0], (len(close_test[:,0]),1))
	test329 = np.reshape(close_test[:,1], (len(close_test[:,1]),1))
	Xtest292, Ytest292 = getFeatures(test292)
	Xtest329, Ytest329 = getFeatures(test329)
	
	rf = RandomForestRegressor(n_estimators=60)
	rf.fit(Xtrain, Ytrain)
	Y_pred292 = rf.predict(Xtest292)
	Y_pred329 = rf.predict(Xtest329)
	#print "Y_pred292.shape",Y_pred292.shape
	#print "Y_pred329.shape",Y_pred329.shape
	Y_292 = np.zeros((len(Y_pred292)))
	Y_329 = np.zeros((len(Y_pred329)))
	
	#Must normlize either before or after predicted values
	for l in range(0, len(Y_292)):
		Y_292[l] = Y_pred292[l] + test292[l+105,0]
		
	for a in range(0, len(Y_329)):
		Y_329[a] = Y_pred329[a] + test329[a+105,0]
	
	f1 = Xtest292[0:200,0]
	f2 = Xtest292[0:200,1]
	f3 = Xtest292[0:200,2]
	f4 = Xtest292[0:200,3]

	#Y_292 = np.reshape(Y_292, (len(Y_292),1))
	Ytest2 = np.zeros((len(Y_292)))
	Ytest2[0:len(Y_292)] = test292[105:,0]
	Y292first = np.zeros(200)
	Y292first[105:200] = Y_292[0:95]
	Y292last = Y_292[len(Y_292)-200:]
	Ytest3 = np.zeros((len(Y_329)))
	Ytest3[0:len(Y_329)] = test329[105:,0]
	Y329first = np.zeros(200)
	Y329first[105:200] = Y_329[0:95]
	Y329last = Y_329[len(Y_329)-200:]
	
	sum_val1 = 0.0
	for i in range(len(Y_292)):
		sum_val1 += ((Y_292[i] - Ytest2[i])**2)
	rms2921 = math.sqrt(sum_val1/len(Ytest2))
	coeff2921 = np.corrcoef(Y_292,Ytest2)[0,1]
	print "RMS value for ML4T-292 data: ", rms2921
	print "Correlation value for ML4T-292 data: ", coeff2921

	sum_val = 0.0
	for j in range(len(Y_329)):
		sum_val += ((Y_329[j] - Ytest3[j])**2)
	rms329 = math.sqrt(sum_val/len(Ytest3))
	coeff329 = np.corrcoef(Y_329,Ytest3)[0,1]
	print "RMS value for ML4T-329 data: ", rms329
	print "Correlation value for ML4T-329 data: ", coeff329
	
	days = range(1,201)
	
	plt.clf()
	subplot(3, 1, 1)
	plt.title('ML4T-292 data: First 200 days')
	plt.ylabel('Y value')
	plt.xlabel('Days')
	plt.plot(days, Ytest2[0:200], label="Y_Actual", color ='b')
	plt.plot(days, Y292first, label="Y_Predicted", color ='r')
	plt.legend(loc='lower left')
	
	subplot(3, 1, 3)
	plt.title('ML4T-329 data: First 200 days')
	plt.ylabel('Y value')
	plt.xlabel('Days')
	plt.plot(days, Ytest3[0:200], label="Y_Actual", color ='b')
	plt.plot(days, Y329first, label="Y_Predicted", color ='r')
	plt.legend(loc='lower left')
	plt.savefig('Y values for first 200 days-final.png')
	
	plt.clf()
	subplot(3, 1, 1)
	plt.title('ML4T-292 data: Last 200 days')
	plt.ylabel('Y value')
	plt.xlabel('Days')
	plt.plot(days, Ytest2[(len(Ytest2)-200):], label="Y_Actual", color ='b')
	#plt.plot(days, Y_292[(len(Y_292)-200):len(Y_292)], label="Y_Predicted", color ='r')
	plt.plot(days, Y292last, label="Y_Predicted", color ='r')
	plt.legend(loc='lower left')
	
	subplot(3, 1, 3)
	plt.title('ML4T-329 data: Last 200 days')
	plt.ylabel('Y value')
	plt.xlabel('Days')
	plt.plot(days, Ytest3[0:200], label="Y_Actual", color ='b')
	#plt.plot(days, Y_329[(len(Y_329)-200):len(Y_329)], label="Y_Predicted", color ='r')
	plt.plot(days, Y329last, label="Y_Predicted", color ='r')
	plt.legend(loc='lower left')
	plt.savefig('Y values for last 200 days-final.png')
	
	plt.clf()
	subplot(3, 1, 1)
	plt.scatter(Y_292, Ytest2)
	plt.xlabel('Predicted Y')
	plt.ylabel('Actual Y')
	plt.title('ML4T-292 data')

	subplot(3, 1, 3)
	plt.scatter(Y_329, Ytest3)
	plt.xlabel('Predicted Y')
	plt.ylabel('Actual Y')
	plt.title('ML4T-329 data')
	plt.savefig('predicted_Y_versus_actual_Y-final.png')
	
	plt.clf()
	plt.title('ML4T-292 features for first 200 days')
	plt.ylabel('Features')
	plt.xlabel('Days')
	plt.plot(days,f1, label='f1')
	plt.plot(days,f2, label='f2')
	plt.plot(days,f3, label='f3')
	plt.plot(days,f4, label='f4')
	plt.legend()
	plt.savefig('ML4T-292 features-fc2-final.png')

if __name__ == '__main__':
	main()