import csv
import pandas as pd


def load_dataset(dataset_name):
    files_folder = dataset_name+'/'
    dataSetIn = files_folder + 'Export_textiles.tsv'
    # dataSetIn = files_folder + 'diag.tsv'
    # dataSetIn = files_folder + 'dd.tsv'

    # print dataSetIn
    with open(dataSetIn) as input:
        dataArray = [x.strip().split('\t') for x in input]	# store the file.
    # print dataArray 						# print entire dataset
#    print len(dataArray)
    binaryData = [['' for i in range (0,4)]for i in range (0,(len(dataArray)-1)*10)]	# declaring the List for binary dataset
    x = 0
    for i in range(1,len(dataArray),):
        if dataArray[i][1] == 'Severely_bad':
#	    print dataArray[x][1]
	    binaryData[x][0] = dataArray[i][1]
	elif dataArray[i][1] == 'Same':
#	    print dataArray[x][1]
	    binaryData[x][0] = dataArray[i][1]
	elif dataArray[i][1] == 'Slightly_worse':
#	    print dataArray[x][1]
	    binaryData[x][0] = dataArray[i][1]
	else:
	    print "Error : Check the TSV file for Economical State"

        if dataArray[i][2] == 'After_6_mos':
#	    print dataArray[x][2]
	    binaryData[x][1] = dataArray[i][2]
	elif dataArray[i][2] == 'After_12_mos':
#	    print dataArray[x][2]
	    binaryData[x][1] = dataArray[i][2]
	elif dataArray[i][2] == 'Now':
#	    print dataArray[x][2]
	    binaryData[x][1] = dataArray[i][2]
	else:
	    print "Error : Check the TSV file for Decision"


        if dataArray[i][3] == '9.005e5':
	    binaryData[x][2] = 0.47
	    binaryData[x][3] = dataArray[i][3]
        elif dataArray[i][3] == '-7.11e5':
	    binaryData[x][2] = 0
	    binaryData[x][3] = dataArray[i][3]
        elif dataArray[i][3] == '7.66e5':
	    binaryData[x][2] = 0.42
	    binaryData[x][3] = dataArray[i][3]
        elif dataArray[i][3] == '2.726e6':
	    binaryData[x][2] = 1
	    binaryData[x][3] = dataArray[i][3]
        elif dataArray[i][3] == '1.6945e6':
	    binaryData[x][2] = 0.69
	    binaryData[x][3] = dataArray[i][3]
        elif dataArray[i][3] == '1.87e6':
	    binaryData[x][2] = 0.75
	    binaryData[x][3] = dataArray[i][3]
        elif dataArray[i][3] == '1.425e6':
	    binaryData[x][2] = 0.62
	    binaryData[x][3] = dataArray[i][3]
        elif dataArray[i][3] == '2.1e5':
	    binaryData[x][2] = 0.27
	    binaryData[x][3] = dataArray[i][3]
        elif dataArray[i][3] == '2.357e6':
	    binaryData[x][2] = 0.89
	    binaryData[x][3] = dataArray[i][3]
        else:
	    print "Reward value "+dataArray[i][3]
        x = x+10

    for i in range(0,(len(binaryData)),10):
        for j in range(0,10):
	    for k in range(0,4):
	        binaryData[i+j][k] = binaryData[i][k]

    for i in range(0,2000,10):
	print binaryData[i]		# print the binary dataset
    print len(binaryData)
#     return binaryData

def get_partial_order(dataset_name):
    if dataset_name == 'Dataset1':
	partialOrder = [['Economical_State'],['Export_Decision']]
	return partialOrder
    else:
	return dataset_name

def get_decNode(dataset_name):
    if dataset_name == 'Dataset1':
	return 'Export_Decision'
    else:
	print "error in Decision var"

def get_utilityNode(dataset_name):
    if dataset_name == 'Dataset1':
	return 'Profit'
    else:
	print "error in Utility Node"

def get_var_set(dataset_name):
    if dataset_name == 'Dataset1':
	partialRest = ['Economical_State','Export_Decision']
	return partialRest
    else:
	print dataset_name

def get_var_col(var_index):
    if var_index == 'Economical_State':
	ar = 0
	return ar
    elif var_index == 'Export_Decision':
	ar = 1
	return ar
    elif var_index == 'Profit':
	ar = 2
	return ar
    else :
	print "Error in partition column name"
	return 3

def get_cluster_length(x):
    if x == 1:
	return 9
    elif x == 2:
	return 8
    elif x == 3:
	return 8
    elif x == 4:
	return 24
    elif x == 5:
	return 8
    elif x == 6:
	return 16




load_dataset('Dataset1')

