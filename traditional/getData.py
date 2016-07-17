# Chitesh Tewani
import numpy as np
import cPickle as pickle
import math
from datetime import datetime
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


def lr_multi_feature2():
    # load the 15-minute time slot
    data_15 = pickle.load(open("../data/60_unnormalized.p"))
    trainPercentage = 0.5
    # get the sorted order of keys
    print len(data_15.keys())
    sorted_dates = data_15.keys()
    sorted_dates.sort()
    datetime_format = '%Y-%m-%d'
    dayOfTheMonthFormat = '%d'
    dayOfTheWeekFormat = '%w'
    zoneCount = 265
    days_in_month = [31, 28, 31, 30, 31, 30]

    # get the training data and test data
    trainData = []
    trainTargetVariable = []
    testTargetVariable = []
    testData = []
    count = 0
    mse = 0
    rmse = 0
    mae = 0
    train_mse = 0
    train_rmse = 0
    train_mae = 0
    lookbackSlotCount = 3
    slot_zone = [[0] * zoneCount]
    for monthIndex in range(len(days_in_month)):
        days = days_in_month[monthIndex]
        # trainCount = int(math.ceil(trainPercentage * days))
        trainCount = days
        testCount = days - trainCount
        # train

        for dayIndex in range(0, trainCount):

            date = datetime.strptime(sorted_dates[count + dayIndex], datetime_format)
            dayOfTheMonth = int(date.strftime(dayOfTheMonthFormat))
            dayOfTheWeek = int(date.strftime(dayOfTheWeekFormat))

            # print len(data_15[sorted_dates[count + dayIndex]])
            for slot in range(len(data_15[sorted_dates[count + dayIndex]])):

                inputParams = [dayOfTheMonth, dayOfTheWeek, slot]
                lookback_slot_zone = slot_zone[-lookbackSlotCount:]
                lookback_slot_zone = zip(*lookback_slot_zone)
                lookback_slot_zone = [sum(lookbackSlot) / len(lookbackSlot) for lookbackSlot in lookback_slot_zone]
                lookback_slot_zone_square = [lookback_slot ** 2 for lookback_slot in lookback_slot_zone]
                lookback_slot_zone_cube = [lookback_slot ** 3 for lookback_slot in lookback_slot_zone]
                # print lookback_slot_zone
                # train_data[0] => input params && train_data[1] => target params
                if monthIndex == len(days_in_month) - 1:
                    testData.append(inputParams + lookback_slot_zone + lookback_slot_zone_square
                                    + lookback_slot_zone_cube)
                    testTargetVariable.append(data_15[sorted_dates[count + dayIndex]][slot])
                else:
                    trainData.append(inputParams + lookback_slot_zone + lookback_slot_zone_square
                                     + lookback_slot_zone_cube)
                    trainTargetVariable.append(data_15[sorted_dates[count + dayIndex]][slot])

                slot_zone.append(data_15[sorted_dates[count + dayIndex]][slot])

        count += days

    print "Training data size", len(trainData), len(trainData[0]), trainData[0], len(trainTargetVariable)
    print "Testing data size", len(testData), len(testTargetVariable)

    # print "Target variable", len(trainTargetVariable), trainTargetVariable[0], data_15['2015-06-30'][95][6]
    for zone in range(zoneCount):
        # print "Testing LR for zone ", zone
        clf = linear_model.LinearRegression()
        targetVariable = [target[zone] for target in trainTargetVariable]
        clf.fit(trainData, targetVariable)

        train_predictedTargetVariable = clf.predict(trainData)
        train_rmse_zone = mean_squared_error(targetVariable, train_predictedTargetVariable)
        train_mse += train_rmse_zone
        train_rmse += train_rmse_zone ** 0.5
        train_mae += mean_absolute_error(targetVariable, train_predictedTargetVariable)

        predictedTargetVariable = clf.predict(testData)

        targetVariable = [target[zone] for target in testTargetVariable]
        rmse_zone = mean_squared_error(targetVariable, predictedTargetVariable)
        mse += rmse_zone
        rmse += rmse_zone ** 0.5
        mae += mean_absolute_error(targetVariable, predictedTargetVariable)

    print "### TRAIN ###"
    print "Mean Squared Error ", round(train_mse, 2), " mean ", round(train_mse / zoneCount, 2)
    print
    print "Root mean squared error ", round(train_rmse, 2), " mean ", round(train_rmse / zoneCount, 2)
    print
    print "Mean absolute Error", round(train_mae, 2), " mean ", round(train_mae / zoneCount, 2)
    print

    print "### TEST ###"
    print "Mean Squared Error ", round(mse, 2), " mean ", round(mse / zoneCount, 2)
    print
    print "Root mean squared error ", round(rmse, 2), " mean ", round(rmse / zoneCount, 2)
    print
    print "Mean absolute Error", round(mae, 2), " mean ", round(mae / zoneCount, 2)


def lr_multi_feature():
    # load the 15-minute time slot
    data_15 = pickle.load(open("../data/15_unnormalized.p"))
    trainPercentage = 0.5
    # get the sorted order of keys
    print len(data_15.keys())
    sorted_dates = data_15.keys()
    sorted_dates.sort()
    datetime_format = '%Y-%m-%d'
    dayOfTheMonthFormat = '%d'
    dayOfTheWeekFormat = '%w'
    zoneCount = 265
    days_in_month = [31, 28, 31, 30, 31, 30]

    # get the training data and test data
    trainData = []
    trainTargetVariable = []
    testTargetVariable = []
    testData = []
    count = 0
    mse = 0
    rmse = 0
    mae = 0
    train_mse = 0
    train_rmse = 0
    train_mae = 0
    lookbackSlotCount = 3
    slot_zone = [[0] * zoneCount]
    for monthIndex in range(len(days_in_month)):
        days = days_in_month[monthIndex]
        # trainCount = int(math.ceil(trainPercentage * days))
        trainCount = days
        testCount = days - trainCount
        # train


        for dayIndex in range(0, trainCount):

            date = datetime.strptime(sorted_dates[count + dayIndex], datetime_format)
            dayOfTheMonth = int(date.strftime(dayOfTheMonthFormat))
            dayOfTheWeek = int(date.strftime(dayOfTheWeekFormat))

            # print len(data_15[sorted_dates[count + dayIndex]])
            for slot in range(len(data_15[sorted_dates[count + dayIndex]])):

                inputParams = [dayOfTheMonth, dayOfTheWeek, slot]
                lookback_slot_zone = slot_zone[-lookbackSlotCount:]
                lookback_slot_zone = zip(*lookback_slot_zone)
                lookback_slot_zone = [sum(lookbackSlot) / len(lookbackSlot) for lookbackSlot in lookback_slot_zone]
                # print lookback_slot_zone
                # train_data[0] => input params && train_data[1] => target params
                if monthIndex == len(days_in_month) - 1:
                    testData.append(inputParams + lookback_slot_zone)
                    testTargetVariable.append(data_15[sorted_dates[count + dayIndex]][slot])
                else:
                    trainData.append(inputParams + lookback_slot_zone)
                    trainTargetVariable.append(data_15[sorted_dates[count + dayIndex]][slot])

                slot_zone.append(data_15[sorted_dates[count + dayIndex]][slot])

        count += days

    print "Training data size", len(trainData), len(trainData[0]), trainData[0], len(trainTargetVariable)
    print "Testing data size", len(testData), len(testTargetVariable)

    # print "Target variable", len(trainTargetVariable), trainTargetVariable[0], data_15['2015-06-30'][95][6]
    for zone in range(zoneCount):
        # print "Testing LR for zone ", zone
        clf = linear_model.LinearRegression()
        targetVariable = [target[zone] for target in trainTargetVariable]
        clf.fit(trainData, targetVariable)

        train_predictedTargetVariable = clf.predict(trainData)
        train_rmse_zone = mean_squared_error(targetVariable, train_predictedTargetVariable)
        train_mse += train_rmse_zone
        train_rmse += train_rmse_zone ** 0.5
        train_mae += mean_absolute_error(targetVariable, train_predictedTargetVariable)

        predictedTargetVariable = clf.predict(testData)

        targetVariable = [target[zone] for target in testTargetVariable]
        rmse_zone = mean_squared_error(targetVariable, predictedTargetVariable)
        mse += rmse_zone
        rmse += rmse_zone ** 0.5
        mae += mean_absolute_error(targetVariable, predictedTargetVariable)

    print "### TRAIN ###"
    print "Mean Squared Error ", train_mse, " mean ", train_mse / zoneCount
    print
    print "Root mean squared error ", train_rmse, " mean ", train_rmse / zoneCount
    print
    print "Mean absolute Error", train_mae, " mean ", train_mae / zoneCount
    print

    print "### TEST ###"
    print "Mean Squared Error ", mse, " mean ", mse / zoneCount
    print
    print "Root mean squared error ", " mean ", rmse, rmse / zoneCount
    print
    print "Mean absolute Error", mae, " mean ", mae / zoneCount


def ridge_multi_feature():
    # load the 15-minute time slot
    data_15 = pickle.load(open("../data/60_unnormalized.p"))
    trainPercentage = 0.5
    # get the sorted order of keys
    print len(data_15.keys())
    sorted_dates = data_15.keys()
    sorted_dates.sort()
    datetime_format = '%Y-%m-%d'
    dayOfTheMonthFormat = '%d'
    dayOfTheWeekFormat = '%w'
    zoneCount = 265
    days_in_month = [31, 28, 31, 30, 31, 30]

    # get the training data and test data
    trainData = []
    trainTargetVariable = []
    testTargetVariable = []
    testData = []
    count = 0
    lookbackSlotCount = 3
    slot_zone = [[0] * zoneCount]
    for monthIndex in range(len(days_in_month)):
        days = days_in_month[monthIndex]
        # trainCount = int(math.ceil(trainPercentage * days))
        trainCount = days
        testCount = days - trainCount
        # train


        for dayIndex in range(0, trainCount):

            date = datetime.strptime(sorted_dates[count + dayIndex], datetime_format)
            dayOfTheMonth = int(date.strftime(dayOfTheMonthFormat))
            dayOfTheWeek = int(date.strftime(dayOfTheWeekFormat))

            # print len(data_15[sorted_dates[count + dayIndex]])
            for slot in range(len(data_15[sorted_dates[count + dayIndex]])):

                inputParams = [dayOfTheMonth, dayOfTheWeek, slot]
                lookback_slot_zone = slot_zone[-lookbackSlotCount:]
                lookback_slot_zone = zip(*lookback_slot_zone)
                lookback_slot_zone = [sum(lookbackSlot) / len(lookbackSlot) for lookbackSlot in lookback_slot_zone]
                # print lookback_slot_zone
                # train_data[0] => input params && train_data[1] => target params
                if monthIndex == len(days_in_month) - 1:
                    testData.append(inputParams + lookback_slot_zone)
                    testTargetVariable.append(data_15[sorted_dates[count + dayIndex]][slot])
                else:
                    trainData.append(inputParams + lookback_slot_zone)
                    trainTargetVariable.append(data_15[sorted_dates[count + dayIndex]][slot])

                slot_zone.append(data_15[sorted_dates[count + dayIndex]][slot])

        count += days

    print "Training data size", len(trainData), len(trainTargetVariable)
    print "Testing data size", len(testData), len(testTargetVariable)

    # print "Target variable", len(trainTargetVariable), trainTargetVariable[0], data_15['2015-06-30'][95][6]
    alphas = [0.25, 0.27, 0.325, 0.35, 0.55, 0.6, 0.7]
    for alpha in alphas:
        mse = 0
        rmse = 0
        mae = 0
        train_mse = 0
        train_rmse = 0
        train_mae = 0
        for zone in range(zoneCount):
            # print "Testing LR for zone ", zone
            clf = linear_model.Ridge(alpha)
            targetVariable = [target[zone] for target in trainTargetVariable]
            clf.fit(trainData, targetVariable)

            train_predictedTargetVariable = clf.predict(trainData)
            train_rmse_zone = mean_squared_error(targetVariable, train_predictedTargetVariable)
            train_mse += train_rmse_zone
            train_rmse += train_rmse_zone ** 0.5
            train_mae += mean_absolute_error(targetVariable, train_predictedTargetVariable)

            predictedTargetVariable = clf.predict(testData)

            targetVariable = [target[zone] for target in testTargetVariable]
            rmse_zone = mean_squared_error(targetVariable, predictedTargetVariable)
            mse += rmse_zone
            rmse += rmse_zone ** 0.5
            mae += mean_absolute_error(targetVariable, predictedTargetVariable)

        print "%%%% ALPHA %%%", alpha
        print "### TRAIN ###"
        print "Mean Squared Error ", train_mse, " mean ", train_mse / zoneCount
        print
        print "Root mean squared error ", train_rmse, " mean ", train_rmse / zoneCount
        print
        print "Mean absolute Error", train_mae, " mean ", train_mae / zoneCount
        print

        print "### TEST ###"
        print "Mean Squared Error ", mse, " mean ", mse / zoneCount
        print
        print "Root mean squared error ", " mean ", rmse, rmse / zoneCount
        print
        print "Mean absolute Error", mae, " mean ", mae / zoneCount


def lasso_multi_feature():
    # load the 15-minute time slot
    data_15 = pickle.load(open("/home/jerry/workspace/uber_nyc_data/data/30_unnormalized.p"))
    slotCount = 48
    trainPercentage = 0.5
    # get the sorted order of keys
    print len(data_15.keys())
    sorted_dates = data_15.keys()
    sorted_dates.sort()
    datetime_format = '%Y-%m-%d'
    dayOfTheMonthFormat = '%d'
    dayOfTheWeekFormat = '%w'
    zoneCount = 265
    days_in_month = [31, 28, 31, 30, 31, 30]

    # get the training data and test data
    trainData = []
    trainTargetVariable = []
    testTargetVariable = []
    testData = []
    count = 0
    lookbackSlotCount = 3
    slot_zone = [[0] * zoneCount]
    for monthIndex in range(len(days_in_month)):
        days = days_in_month[monthIndex]
        # trainCount = int(math.ceil(trainPercentage * days))
        trainCount = days
        testCount = days - trainCount
        # train


        for dayIndex in range(0, trainCount):

            date = datetime.strptime(sorted_dates[count + dayIndex], datetime_format)
            dayOfTheMonth = int(date.strftime(dayOfTheMonthFormat))
            dayOfTheWeek = int(date.strftime(dayOfTheWeekFormat))

            # print len(data_15[sorted_dates[count + dayIndex]])
            for slot in range(len(data_15[sorted_dates[count + dayIndex]])):

                inputParams = [dayOfTheMonth, dayOfTheWeek, slot]
                lookback_slot_zone = slot_zone[-lookbackSlotCount:]
                lookback_slot_zone = zip(*lookback_slot_zone)
                lookback_slot_zone = [sum(lookbackSlot) / len(lookbackSlot) for lookbackSlot in lookback_slot_zone]
                # print lookback_slot_zone
                # train_data[0] => input params && train_data[1] => target params
                if monthIndex == len(days_in_month) - 1:
                    testData.append(inputParams + lookback_slot_zone)
                    testTargetVariable.append(data_15[sorted_dates[count + dayIndex]][slot])
                else:
                    trainData.append(inputParams + lookback_slot_zone)
                    trainTargetVariable.append(data_15[sorted_dates[count + dayIndex]][slot])

                slot_zone.append(data_15[sorted_dates[count + dayIndex]][slot])

        count += days

    print "Training data size", len(trainData), len(trainTargetVariable)
    print "Testing data size", len(testData), len(testTargetVariable)

    # print "Target variable", len(trainTargetVariable), trainTargetVariable[0], data_15['2015-06-30'][95][6]
    # alphas = [0.25,0.27,0.325,0.35,0.55,0.6,0.7]
    alphas = [0.7]
    for alpha in alphas:
        mse = 0
        rmse = 0
        mae = 0
        train_mse = 0
        train_rmse = 0
        train_mae = 0
        final_prediction = []
        ground_truth = []
        for i in xrange(slotCount * 30):
            final_prediction.append([])
            ground_truth.append([])
        for zone in range(zoneCount):
            # print "Testing LR for zone ", zone
            # clf = linear_model.Lasso(alpha)
            clf = linear_model.Lasso(alpha)
            targetVariable = [target[zone] for target in trainTargetVariable]
            clf.fit(trainData, targetVariable)

            train_predictedTargetVariable = clf.predict(trainData)
            train_rmse_zone = mean_squared_error(targetVariable, train_predictedTargetVariable)
            train_mse += train_rmse_zone
            train_rmse += train_rmse_zone ** 0.5
            train_mae += mean_absolute_error(targetVariable, train_predictedTargetVariable)

            predictedTargetVariable = clf.predict(testData)

            for slotNumber in xrange(slotCount):
                predict = predictedTargetVariable[slotNumber]
                final_prediction[slotNumber].append(predict)

            targetVariable = [target[zone] for target in testTargetVariable]

            for slotNumber in xrange(slotCount):
                expected = targetVariable[slotNumber]
                ground_truth[slotNumber].append(expected)

            rmse_zone = mean_squared_error(targetVariable, predictedTargetVariable)
            mse += rmse_zone
            rmse += rmse_zone ** 0.5
            mae += mean_absolute_error(targetVariable, predictedTargetVariable)

        print "%%%% ALPHA %%%", alpha
        print "### TRAIN ###"
        print "Mean Squared Error ", train_mse, " mean ", train_mse / zoneCount
        print
        print "Root mean squared error ", train_rmse, " mean ", train_rmse / zoneCount
        print
        print "Mean absolute Error", train_mae, " mean ", train_mae / zoneCount
        print

        print "### TEST ###"
        print "Mean Squared Error ", mse, " mean ", mse / zoneCount
        print
        print "Root mean squared error ", " mean ", rmse, rmse / zoneCount
        print
        print "Mean absolute Error", mae, " mean ", mae / zoneCount

        prediction_dates = sorted_dates[-30:]
        iterIndex = 0
        for date in prediction_dates:
            for slotNumber in xrange(slotCount):
                final_prediction[iterIndex].insert(0, date)
                final_prediction[iterIndex].insert(0, slotNumber)

                ground_truth[iterIndex].insert(0, date)
                ground_truth[iterIndex].insert(0, slotNumber)

                iterIndex += 1

    np.savetxt('final_prediction.txt', np.array(final_prediction), delimiter=',', fmt='%s')
    np.savetxt('ground_truth.txt', np.array(ground_truth), delimiter=',', fmt='%s')


def main():
    # load the 15-minute time slot
    data_15 = pickle.load(open("../data/15_unnormalized.p"))
    trainPercentage = 0.5
    # get the sorted order of keys
    print len(data_15.keys())
    sorted_dates = data_15.keys()
    sorted_dates.sort()
    datetime_format = '%Y-%m-%d'
    dayOfTheMonthFormat = '%d'
    dayOfTheWeekFormat = '%w'
    zoneCount = 265
    days_in_month = [31, 28, 31, 30, 31, 30]

    # get the training data and test data
    trainData = []
    trainTargetVariable = []
    testTargetVariable = []
    testData = []
    count = 0
    mse = 0
    rmse = 0
    mae = 0
    train_mse = 0
    train_rmse = 0
    train_mae = 0
    for monthIndex in range(len(days_in_month)):
        days = days_in_month[monthIndex]
        # trainCount = int(math.ceil(trainPercentage * days))
        trainCount = days
        testCount = days - trainCount
        # train

        for dayIndex in range(0, trainCount):

            date = datetime.strptime(sorted_dates[count + dayIndex], datetime_format)
            dayOfTheMonth = int(date.strftime(dayOfTheMonthFormat))
            dayOfTheWeek = int(date.strftime(dayOfTheWeekFormat))

            # print len(data_15[sorted_dates[count + dayIndex]])
            for slot in range(len(data_15[sorted_dates[count + dayIndex]])):

                inputParams = [dayOfTheMonth, dayOfTheWeek, slot]
                # train_data[0] => input params && train_data[1] => target params
                if monthIndex == len(days_in_month) - 1:
                    testData.append(inputParams)
                    testTargetVariable.append(data_15[sorted_dates[count + dayIndex]][slot])
                else:
                    trainData.append(inputParams)
                    trainTargetVariable.append(data_15[sorted_dates[count + dayIndex]][slot])

        count += days

    print "Training data size", len(trainData), len(trainTargetVariable)
    print "Testing data size", len(testData), len(testTargetVariable)

    # print "Target variable", len(trainTargetVariable), trainTargetVariable[0], data_15['2015-06-30'][95][6]
    for zone in range(zoneCount):
        # print "Testing LR for zone ", zone
        clf = linear_model.LinearRegression()
        targetVariable = [target[zone] for target in trainTargetVariable]
        clf.fit(trainData, targetVariable)

        train_predictedTargetVariable = clf.predict(trainData)
        train_rmse_zone = mean_squared_error(targetVariable, train_predictedTargetVariable)
        train_mse += train_rmse_zone
        train_rmse += train_rmse_zone ** 0.5
        train_mae += mean_absolute_error(targetVariable, train_predictedTargetVariable)

        predictedTargetVariable = clf.predict(testData)

        targetVariable = [target[zone] for target in testTargetVariable]
        rmse_zone = mean_squared_error(targetVariable, predictedTargetVariable)
        mse += rmse_zone
        rmse += rmse_zone ** 0.5
        mae += mean_absolute_error(targetVariable, predictedTargetVariable)

    print "### TRAIN ###"
    print "Mean Squared Error ", train_mse, " mean ", train_mse / zoneCount
    print
    print "Root mean squared error ", train_rmse, " mean ", train_rmse / zoneCount
    print
    print "Mean absolute Error", train_mae, " mean ", train_mae / zoneCount
    print

    print "### TEST ###"
    print "Mean Squared Error ", mse, " mean ", mse / zoneCount
    print
    print "Root mean squared error ", " mean ", rmse, rmse / zoneCount
    print
    print "Mean absolute Error", mae, " mean ", mae / zoneCount


if __name__ == '__main__':
    # main()
    # lr_multi_feature()
    # lr_multi_feature2()
    # ridge_multi_feature()
    lasso_multi_feature()
