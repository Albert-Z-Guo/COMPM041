# COMPM041
# Student Name: Zunran Guo
# Student Number: 15118320
# All work is original.

import time
import pandas as pd
import numpy as np

"""
binarize_attribute(base_filename, fit_filename, base_file_attribute_column_index, split_indicator)
is the method that binarize a attribute based on the dimension of the attribute. If the dimension
of the attribute in the base file is not the same as the dimension of the attribute in the fit file,
the larger dimension will be chosen in the preprosseing procedure. In addition, the method will also 
adjust the attribute column index accordingly, as the same attribute has different column indices in
the base file and the fit file.
"""
def binarize_attribute(base_filename, fit_filename, base_file_attribute_column_index, split_indicator):
    # configure attribute column indices
    if (base_filename == 'data_train.txt'):
        if (fit_filename == 'data_train.txt'):
            fit_file_attribute_column_index = base_file_attribute_column_index
        else:
            fit_file_attribute_column_index = base_file_attribute_column_index - 1
    elif (base_filename == 'shuffle_data_test.txt'):
        if (fit_filename == 'data_train.txt'):
            fit_file_attribute_column_index = base_file_attribute_column_index + 1
        else:
            fit_file_attribute_column_index = base_file_attribute_column_index

    # append all split tags to attributes
    attributes = []
    
    with open(base_filename, 'r') as infile:
        if (split_indicator == 1):
            for line in infile:
                line_split = line.strip().split()[base_file_attribute_column_index].split(',') 
                for i in range(len(line_split)):
                    attributes.append(line_split[i])
        else:
            for line in infile:
                attributes.append(line.strip().split()[base_file_attribute_column_index])
    
    # set attributes
    attributes_set_raw = set(attributes)
    
    # cast attributes_set_raw to indexable attributes_set
    attributes_set = []
    for i in attributes_set_raw:
        attributes_set.append(i)

    # testing
#     attributes_set.sort()
#     for i in range(0, len(attributes_set)):
#         print("%d: %s " % (i + 1, attributes_set[i]))
    
    # define a method to check whether 'entry' is in 'array'
    def check(array, entry):
        for i in range(0, len(array)):
            # number or string compare
            if (array[i] == entry):
                return i + 1
        return False
    
    # fill in boolean values
    index = 0
    number_of_datapoints = sum(1 for line in open(fit_filename))
    attributes_binarized = np.zeros((number_of_datapoints, len(attributes_set)))
    
    with open(fit_filename, 'r') as infile:
        if (split_indicator == 1):
            for line in infile:
                line_split = line.strip().split()[fit_file_attribute_column_index].split(',')
                for i in range(0, len(line_split)):
                    result = check(attributes_set, line_split[i])
                    if (result != False):
                        attributes_binarized[index][result - 1] = 1
                index += 1
        else:
            for line in infile: 
                entry = line.strip().split()[fit_file_attribute_column_index]
                result = check(attributes_set, entry)
                if (result != False):
                    attributes_binarized[index][result - 1] = 1
                index += 1
    return attributes_binarized
 
 
"""
binarize_price(filename, attribute_column_index, training_file_indicator)
is the method that binarizes Ad slot floor price (RMB/CPM) in 5 dimensions,
which correspond to the 5 sections of the price range:
0, [1,10], [11,50], [51,100] and [101, +infinity]
"""
def binarize_price(filename, attribute_column_index, training_file_indicator):
    # create customized binarized form
    number_of_datapoints = sum(1 for line in open(filename))

    # split the price range to 5 sections
    # 0, [1,10], [11,50], [51,100] and [101, +infinity]
    price_binarized = np.zeros((number_of_datapoints, 5))

    # read file
    with open(filename, 'r') as infile:
        # check whether test file is opened
        if (training_file_indicator == 0):
            attribute_column_index -= 1 
        
        # read line by line
        index = 0
        for line in infile:
            price = int(line.strip().split()[attribute_column_index])

            # condition judgement
            if (price == 0):
                price_binarized[index][0] = 1
            
            elif (price >= 1 and price <= 10):
                price_binarized[index][1] = 1
            
            elif (price >= 11 and price <= 50):
                price_binarized[index][2] = 1
            
            elif (price >= 51 and price <= 100):
                price_binarized[index][3] = 1
            
            elif (price >= 101):
                price_binarized[index][4] = 1

            # update index
            index += 1
    return price_binarized

def precondition(filename, train_indicator):
    # load data
    start = time.time()  # start timing
    print("loading and preconditioning data...")
    datapoints = pd.read_csv(filename, sep='\t', header=-1)
    print 'initial data points shape:', datapoints.shape
    
    # check train_indicator
    if (train_indicator == 1):
        # extract the 'Click' column
        y = datapoints.ix[:, 0]
        datapoints = datapoints.drop(datapoints.columns[[0]], axis=1)
        
    # add column names to the data frame
    datapoints.columns = ['Weekday', 'Hour', 'Timestamp', 'Log Type', \
                          'UserID', 'User-Agent', 'IP', 'Region', 'City', 'Ad Exchange', \
                         'Domain', 'URL', 'Anonymous URL ID', 'Ad slot ID', 'Ad slot width', \
                         'Ad slot height', 'Ad slot visibility', 'Ad slot format', \
                         'Ad slot floor price (RMB/CPM)', 'Creative ID', 'Key Page URL', \
                         'Advertiser ID', 'User Tags']
   
    # binarize attributes
    Weekdays = pd.DataFrame(binarize_attribute('data_train.txt', filename, 1, 0))
    Hours = pd.get_dummies(datapoints['Hour'])
    UserAgents = pd.DataFrame(binarize_attribute('shuffle_data_test.txt', filename, 5, 0))
    Regions = pd.get_dummies(datapoints['Region'])
    Cities = pd.get_dummies(datapoints['City'])
    AdExchange = pd.get_dummies(datapoints['Ad Exchange'])
    AdSlotWidths = pd.get_dummies(datapoints['Ad slot width'])
    AdSlotHeights = pd.get_dummies(datapoints['Ad slot height'])
    AdSlotAreas = pd.get_dummies(datapoints['Ad slot width'] * datapoints['Ad slot height'])
    AdSlotVisibility = pd.get_dummies(datapoints['Ad slot visibility'])
    AdSlotFormats = pd.get_dummies(datapoints['Ad slot format'])
    CreativeID = pd.get_dummies(datapoints['Creative ID'])
    UserTags = pd.DataFrame(binarize_attribute('shuffle_data_test.txt', filename, 22, 1))
    AdSlotFloorPrices = pd.DataFrame(binarize_price(filename, 19, train_indicator))
    
    print("\nWeekdays has          %d      columns of binarized attributes." % (len(Weekdays.columns)))
    print("Hours has             %d     columns of binarized attributes." % (len(Hours.columns)))
    print("UserAgents has        %d     columns of binarized attributes." % (len(UserAgents.columns)))
    print("Regions has           %d     columns of binarized attributes." % (len(Regions.columns)))
    print("Cities has            %d    columns of binarized attributes." % (len(Cities.columns)))
    print("AdExchange has        %d      columns of binarized attributes." % (len(AdExchange.columns)))
    print("AdSlotWidth has       %d     columns of binarized attributes." % (len(AdSlotWidths.columns)))
    print("AdSlotHeight has      %d      columns of binarized attributes." % (len(AdSlotHeights.columns)))
    print("AdSlotAreas has       %d     columns of binarized attributes." % (len(AdSlotAreas.columns)))
    print("AdSlotVisibility has  %d      columns of binarized attributes." % (len(AdSlotVisibility.columns)))
    print("AdSlotFormats has     %d      columns of binarized attributes." % (len(AdSlotFormats.columns)))
    print("AdSlotFloorPrices has %d      columns of binarized attributes." % (len(AdSlotFloorPrices.columns)))
    print("CreativeID has        %d     columns of binarized attributes." % (len(CreativeID.columns)))
    print("UserTags has          %d     columns of binarized attributes." % (len(UserTags.columns)))
    
    # concatenate all the binarized attributes
    binarized = pd.concat([Weekdays, Hours, UserAgents, Regions, Cities, AdExchange, \
                           AdSlotWidths, AdSlotHeights, AdSlotVisibility, AdSlotFormats, CreativeID,\
                           AdSlotFloorPrices, UserTags], axis=1)     
         
    # normalize certain attribute
    datapoints['Ad slot floor price (RMB/CPM)'] = (datapoints['Ad slot floor price (RMB/CPM)'] - datapoints['Ad slot floor price (RMB/CPM)'].mean()) / (datapoints['Ad slot floor price (RMB/CPM)'].std())
    
    # concatenate the normalized attributes and the binarized attributes
    X = pd.concat([datapoints['Ad slot floor price (RMB/CPM)'], binarized], axis=1)
    
    # shuffle all the datapoints 
#     X = X.reindex(np.random.permutation(X.index))
    
    end = time.time()  # end timing
    print("\ntotal preprocessing time: %d seconds" % (end - start))
    
    if (train_indicator == 1):
        print 'X shape:', X.shape
        print 'y shape:', y.shape
        return X, y
    
    if (train_indicator == 0):
        print 'X shape:', X.shape
        return X
    
def sample(X, y):
    start = time.time()  # start timing
    print("\nsampling...")
    
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    
    # user may choose save to comma-separated values files without headers and indices
#     X_train.to_csv('X_train_medium.csv', header=False, index=False)
#     y_train.to_csv('y_train_medium.csv', header=False, index=False)
#     X_test.to_csv('X_test.csv', header=False, index=False)
#     y_test.to_csv('y_test.csv', header=False, index=False)
    
    end = time.time()  # end timing
    print("total sampling time: %d seconds" % (end - start))
    
    print 'X_train shape:', X_train.shape
    print 'y_train shape:', y_train.shape
    print 'X_test shape:', X_test.shape
    print 'y_test shape:', y_test.shape
    
    return X_train, X_test, y_train, y_test
    
    
    
    
