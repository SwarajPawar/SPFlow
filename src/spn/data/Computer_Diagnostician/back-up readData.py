import csv
import pandas as pd


def load_dataset(dataset_name):
    files_folder = dataset_name+'/'					# load the folder name
    dataSetIn = files_folder + 'Computer_diagnostician.tsv'	# and the main dataset-file
    # dataSetIn = files_folder + 'diag.tsv'				# and the dummy dataset-file (5 rows)
    # dataSetIn = files_folder + 'dd.tsv'				# and the dummy dataset-file (30 rows)

    # print dataSetIn
    with open(dataSetIn) as input:
    #    print zip(*(line.strip().split('\t') for line in input))
        dataArray = [x.strip().split('\t') for x in input]		# store the file in the list format.
    # print dataArray 						# print the entire dataset in list form
#    print len(dataArray)
    init_data_array = dataArray
    low = 125
    med = 225
    hi = 300
    binaryData = [['' for i in range (0,8)]for i in range (0,(len(dataArray)-1)*10)]	# declaring the List for binary dataset
    x = 0
    for i in range(1,len(dataArray),):
# Logic_Board and IO_Board conversion take 1 bits (bit 0)
        if dataArray[i][1] == 'Logic_board':
#	    print dataArray[x][1]
	    binaryData[x][0] = 'Logic_board'
	elif dataArray[i][1] == 'IO_board':
#	    print dataArray[x][1]
	    binaryData[x][0] = 'IO_board'
	else:
	    print "Error : Check the TSV file for Rework_Decision"

# Logic_board_fail conversion takes 1 bit (bit 1)
	if dataArray[i][2] == 'Yes':
#	    print dataArray[x][2]
	    binaryData[x][1] = 'Yes'
	elif dataArray[i][2] == 'No':
#	    print dataArray[x][2]
	    binaryData[x][1] = 'No'
	else:
	    print "Error : Check the TSV file for Logic_board_fail"

# IO_board_fail conversion takes 1 bit (bit 2)
	if dataArray[i][3] == 'Yes':
#	    print dataArray[x][3]
	    binaryData[x][2] = 'Yes'
	elif dataArray[i][3] == 'No':
#	    print dataArray[x][3]
	    binaryData[x][2] = 'No'
	else:
	    print "Error : Check the TSV file for IO_board_fail"

# System_State conversion takes 1 bit (bit 3)
	if dataArray[i][4] == 'Operational':
#	    print dataArray[x][4]
	    binaryData[x][3] = 'Operational'
	elif dataArray[i][4] == 'Failed':
#	    print dataArray[x][4]
	    binaryData[x][3] = 'Failed'
	else:
	    print "Error : Check the TSV file for System_State"

# Rework_Outcome conversion takes 4 bits (bit 4, bit 5, bit 6, bit 7)
	if dataArray[i][5] == 'L0_IO1':
#	    print dataArray[x][5]
	    binaryData[x][4] = 'L0_IO1'

	elif dataArray[i][5] == 'L0_IO0':
#	    print dataArray[x][5]
	    binaryData[x][4] = 'L0_IO0'

	elif dataArray[i][5] == 'L1_IO0':
#	    print dataArray[x][5]
	    binaryData[x][4] = 'L1_IO0'

        elif dataArray[i][5] == 'L1_IO1':
#	    print dataArray[x][5]
	    binaryData[x][4] = 'L1_IO1'

	    print "Error : Check the TSV file for Rework_Outcome"
        else:
	    print "Hi"

        if dataArray[i][6] == '125':
	    binaryData[x][6] = 0	# 125+(-125)/175
	    binaryData[x][7] = dataArray[i][6]
        elif dataArray[i][6] == '175':
	    binaryData[x][6] = 0.3	# 175+(-125)/175
	    binaryData[x][7] = dataArray[i][6]
        elif dataArray[i][6] == '200':
	    binaryData[x][6] = 0.5	# 200+(-125)/175
	    binaryData[x][7] = dataArray[i][6]
        elif dataArray[i][6] == '225':
	    binaryData[x][6] = 0.7	# 225+(-125)/175
	    binaryData[x][7] = dataArray[i][6]
        elif dataArray[i][6] == '300':
	    binaryData[x][6] = 1	# 300+(-125)/175
	    binaryData[x][7] = dataArray[i][6]
        else:
	    print "Reward value "+dataArray[i][6]
        x = x+10

    # print binaryData		# print the binary dataset
    # print len(binaryData)

    for i in range(0,(len(binaryData)),10):
        for j in range(0,10):
	    for k in range(0,8):
	        binaryData[i+j][k] = binaryData[i][k]

    for i in range(0,(len(binaryData)),10):
#        print binaryData[i][9]
        if binaryData[i][6] == 0:
	    for c in range(0,10):
	        binaryData[i+c][5] = 0

        if binaryData[i][6] == 0.3:
	    for c in range(0,7):
	        binaryData[i+c][5] = 0
	    for d in range(7,10):
	        binaryData[i+d][5] = 1

        if binaryData[i][6] == 0.5:
	    for c in range(0,5):
	        binaryData[i+c][5] = 0
	    for d in range(5,10):
	        binaryData[i+d][5] = 1

        if binaryData[i][6] == 0.7:
	    for c in range(0,3):
	        binaryData[i+c][5] = 0
	    for d in range(3,10):
	        binaryData[i+d][5] = 1

        if binaryData[i][6] == 1:
	    for c in range(0,10):
	        binaryData[i+c][5] = 1

#    print len(binaryData)
#    print binaryData
    return binaryData , init_data_array

def get_partial_order(dataset_name):
    if dataset_name == 'Dataset5':
#	partialOrder = [['Logic_board_fail'],['IO_board_fail'],['Rework_Outcome']]
	partialOrder = [['System_State'],['Rework_Decision'],['Logic_board_fail'],['IO_board_fail'],['Rework_Outcome']]
	return partialOrder
    else:
	print dataset_name

def get_decNode(dataset_name):
    if dataset_name == 'Dataset5':
	return 'Rework_Decision'
    else:
	print "error in Dec Node"

def get_utilityNode(dataset_name):
    if dataset_name == 'Dataset5':
	return 'Rework_Outcome'
    else:
	print "error in Dec Node"

def get_var_set(dataset_name):
    if dataset_name == 'Dataset5':
	partialRest = ['System_State','Rework_Decision','Logic_board_fail','IO_board_fail','Rework_Outcome','Rework_Cost']
	return partialRest
    else:
	print dataset_name

def get_var_col(var_index):
    if var_index == 'System_State':
	ar = 3
	return ar
    elif var_index == 'Rework_Decision':
	ar = 0
	return ar
    elif var_index == 'Logic_board_fail':
	ar = 1
	return ar
    elif var_index == 'IO_board_fail':
	ar = 2
	return ar
    elif var_index == 'Rework_Outcome':
        ar = 4
	return ar
    else :
	print "Error in partition column name"
	return 7

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


