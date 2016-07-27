# Tyler Sorg
# Machine Learning
# SVM Project using Spambase

from sklearn.utils import shuffle
from sklearn import svm
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import scipy
import warnings  # Scikit-learn has a lot of deprecation warnings...

warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn import cross_validation


def LoadSpamData(filename="spambase.data"):
    """
    Each line in the datafile is a csv with features values, followed by a
    single label (0 or 1), per sample; one
    sample per line
    """

    "The file function reads the filename from the current directory, unless " \
    "you provide an absolute path e.g. " \
    "/path/to/file/file.py or C:\\path\\to\\file.py"

    unprocessed_data_file = file(filename, 'r')

    "Obtain all lines in the file as a list of strings."

    unprocessed_data = unprocessed_data_file.readlines()

    labels = []
    features = []

    for line in unprocessed_data:
        feature_vector = []

        "Convert the String into a list of strings, being the elements of the " \
        "string separated by commas"
        split_line = line.split(',')

        "Iterate across elements in the split_line except for the final " \
        "element "
        for element in split_line[:-1]:
            feature_vector.append(float(element))

        "Add the new vector of feature values for the sample to the features " \
        "list"
        features.append(feature_vector)

        "Obtain the label for the sample and add it to the labels list"
        labels.append(int(split_line[-1]))

    "Return List of features with its list of corresponding labels"
    return features, labels


def BalanceDataset(features, labels):
    """
    Assumes the lists of features and labels are ordered such that all
    like-labelled samples are together (all the
    zeros come before all the ones, or vice versa)
    """

    count_0 = labels.count(0)
    count_1 = labels.count(1)
    balanced_count = min(count_0, count_1)

    while balanced_count % 20 is not 0:
        balanced_count -= 1

    # >>> a
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # >>> for i in range(len(a)/2+1):
    # ...     print i, a[:i] + a[-i:]
    # ...
    # 0 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # 1 [0, 9]
    # 2 [0, 1, 8, 9]
    # 3 [0, 1, 2, 7, 8, 9]
    # 4 [0, 1, 2, 3, 6, 7, 8, 9]
    # 5 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Indexing with a negative value tracks from the end of the list
    return features[:balanced_count] + features[-balanced_count:], labels[
                                                                   :balanced_count] + labels[
                                                                                      -balanced_count:]


def ConvertDataToArrays(features, labels):
    """
    conversion to a numpy array is easy if you're starting with a List of lists.
    The returned array has dimensions (M,N), where M is the number of lists
    and N is the number of features?

    """

    return np.asarray(features), np.asarray(labels)


def NormalizeFeatures(features):
    """
    I'm providing this mostly as a way to demonstrate array operations using
    Numpy.  Incidentally it also solves a
    small step in the homework.
    """

    "selecting axis=0 causes the mean to be computed across each feature, " \
    "for all the samples"
    # >>> m
    #     array([[1, 2, 3],
    #            [1, 2, 3],
    #            [1, 2, 3]])
    # >>> type(m)
    #     <type 'numpy.ndarray'>
    # >>> np.mean(m,axis=0) # Witchcraft
    #     array([ 1.,  2.,  3.])
    # >>> np.mean(m,axis=1)
    #     array([ 2.,  2.,  2.])
    # >>> np.mean(m,axis=3) # there is no 3rd axis/3rd dimension!

    means = np.mean(features, axis=0)

    variances = np.var(features, axis=0)

    stddevs = np.std(features, axis=0)

    "Operations in numpy performed on a 2D array and a 1D matrix will " \
    "automatically broadcast correctly, " \
    "if the leading dimensions match."
    # features = features - means
    features -= means

    # features /= variances
    features /= stddevs

    return features  # features should now be standardized


def PrintDataToSvmLightFormat(features, labels, filename="svm_features.data"):
    """
    Readable format for SVM Light should be, with
    label 0:feature0, 1:feature1, 2:feature2, etc...
    where label is -1 or 1.
    """

    if len(features) != len(labels):
        raise Exception("Number of samples and labels must match")
    dat_file = file(filename, 'w')
    for s in range(len(features)):

        if labels[s] == 0:
            line = "-1 "
        else:
            line = "1 "

        for f in range(len(features[s])):
            line += "%i:%f " % (f + 1, features[s][f])
        line += "\n"
        dat_file.write(line)
    dat_file.close()


def FeatureSubset(features_array, indices):
    """
    Takes the original set of features and returns a small array containing
    only the features with the given indices.

    features_array is a numpy 2D array of dimension (M,N), where M is the
    number of samples and N is the number of
    features in the feature vector.

    indices are those of the features to be used, as a list of integers
    """
    return features_array[:, indices]


def create_k_subsets_for_training_features_and_labels(training_features,
                                                      training_labels, k):
    # Split training_set into 10 (relatively) equally sized partitions
    tenth_training_set_len = len(training_features) / k
    lower_bound = 0
    upper_bound = tenth_training_set_len
    training_features_subsets = []
    training_labels_subsets = []
    while upper_bound < len(training_features):
        training_features_subsets.append(
            training_features[lower_bound:upper_bound])
        training_labels_subsets.append(training_labels[lower_bound:upper_bound])
        lower_bound = upper_bound
        upper_bound += tenth_training_set_len
    upper_bound -= tenth_training_set_len
    training_features_subsets.append(training_features[upper_bound:])
    training_labels_subsets.append(training_labels[upper_bound:])
    return training_features_subsets, training_labels_subsets


def main():
    # PREPROCESSING DATA

    # GET ALL THE DATA
    features, labels = LoadSpamData()

    # EQUAL POSITIVE AND NEGATIVE EXAMPLES
    features, labels = BalanceDataset(features, labels)

    # SPLIT IN HALF FOR TRAINING AND TESTING
    training_features = []  # Let's say every other feature vector goes in here.
    training_labels = []
    testing_features = []
    testing_labels = []

    for i in range(len(features)):
        if i % 2 is 0:  # put features[i] in training_features, labels[i] in
            # training_labels
            training_features.append(features[i])
            training_labels.append(labels[i])
        else:  # put features[i] in training_features, labels[i] in
            # training_labels
            testing_features.append(features[i])
            testing_labels.append(labels[i])

    # CONVERT TO NUMPY ARRAYS
    training_features, training_labels = ConvertDataToArrays(training_features,
                                                             training_labels)
    testing_features, testing_labels = ConvertDataToArrays(testing_features,
                                                           testing_labels)

    # STANDARDIZE DATA
    # features = NormalizeFeatures(features)

    # Calculate means and standard deviations for each feature in the
    # training set
    training_means = np.mean(training_features, axis=0)
    training_standard_deviations = np.std(training_features, axis=0)

    # Standardize the training set
    training_features -= training_means
    training_features /= training_standard_deviations

    # Standardize the testing set using the training means and standard
    # deviations
    testing_features -= training_means
    testing_features /= training_standard_deviations

    # RANDOMLY SHUFFLE THE TRAINING DATA
    training_features, training_labels = shuffle(training_features,
                                                 training_labels)

    # =========================================================================
    print "\nBeginning experiment 1..\n"
    # 10-fold cross validation to test values of C
    values_of_C = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    training_features_subsets, training_labels_subsets = \
        create_k_subsets_for_training_features_and_labels(
        training_features, training_labels, len(values_of_C))
    # TODO: Turn these into ndarrays. they're lists of ndarrays?


    # total_length_subsets = 0
    # for i in training_features_subsets:
    #     total_length_subsets += len(i)
    # print total_length_subsets, len(training_features)


    classifiers = [svm.SVC(kernel='linear', C=values_of_C[i]) for i in
                   range(len(values_of_C))]

    accuracy_j_array = [0.0 for i in values_of_C]
    confusion_matrices = list()  # [tp, fp, fn, tn] for each iteration of
    # outer loop

    # -------------------------------------------------------------------------
    # K-FOLD CROSS VALIDATION
    # -------------------------------------------------------------------------
    for j in range(len(values_of_C)):  # For each of the 10 values of C to test
        print 'Cross validating classifier%d ten times..' % (j)
        tp, tn, fp, fn = 0, 0, 0, 0
        correct_count_j = 0
        # ---------------------------------------------------------------------
        # IN CASE I DON'T WANT TO USE MY OWN IMPLEMENTATION...
        # The nested for loop recreates the functionality of these two lines:
        # ---------------------------------------------------------------------
        # scores = cross_validation.cross_val_score(classifiers[j],
        # training_features, training_labels, cv=10)
        # accuracy_j_array[j] = np.mean(scores)
        # ---------------------------------------------------------------------

        for i in range(
                10):  # For each of the 10 disjoint subsets of the training set
            # 1) SELECT VALIDATION SUBSET_I
            validation_subset_features_i = training_features_subsets[
                i]  # ith feature subset
            validation_subset_labels_i = training_labels_subsets[
                i]  # ith label subset

            # Select training subsets that aren't index i
            training_subsets_features_all_but_index_i = []
            training_subsets_labels_all_but_index_i = []
            for c in range(len(values_of_C)):
                if c is not i:  # k-1 subsets for each iteration through C
                    # parameter's values
                    training_subsets_features_all_but_index_i.append(
                        training_features_subsets[c])  # list of lists!!
                    training_subsets_labels_all_but_index_i.append(
                        training_labels_subsets[c])  # list of lists!!

            not_validation_features = [feature_array for features_i in
                                       training_subsets_features_all_but_index_i
                                       for
                                       feature_array in features_i]
            not_validation_labels = [label for labels_i in
                                     training_subsets_labels_all_but_index_i for
                                     label in labels_i]

            # 2) TRAIN ON ALL SUBSETS EXCEPT SUBSET_I
            classifiers[j].fit(
                np.array(not_validation_features),
                np.array(not_validation_labels))

            # TODO: Use this instead of the prediction?
            decisions_ji = classifiers[j].decision_function(
                validation_subset_features_i)
            predictions_ji = classifiers[j].predict(
                validation_subset_features_i)

            # 3) TEST LEARNED MODEL ON SUBSET_I TO GET ACCURACY A_JI
            for index in range(len(validation_subset_features_i)):
                actual = validation_subset_labels_i[index]
                prediction = predictions_ji[index]
                if prediction == actual:  # make sure types are the same
                    correct_count_j += 1
                    if actual == 0:
                        tn += 1
                    elif actual == 1:
                        tp += 1
                else:
                    if prediction == 1 and actual == 0:
                        fp += 1
                    elif prediction == 0 and actual == 1:
                        fn += 1

        # Save results of each iteration.
        confusion_matrices.append([tp, fp, fn, tn])
        accuracy_j_array[j] = float(correct_count_j) / 1800

    # -------------------------------------------------------------------------
    # AVERAGE ACCURACIES FOR EACH CLASSIFIER
    # -------------------------------------------------------------------------
    accuracy_j_array = np.array(accuracy_j_array)

    print "\nAverage accuracies: ", accuracy_j_array
    bestC = values_of_C[np.argmax(accuracy_j_array)]
    print "\nLargest accuracy was C#%d = %f" % (
    np.argmax(accuracy_j_array), np.max(accuracy_j_array))

    # TRAIN NEW LINEAR SVM USING BEST C PARAMETER VALUE
    print '\nTraining a new classifier using the best C value (%f)..' % (bestC)
    clf = svm.SVC(kernel='linear', C=bestC)
    clf.fit(training_features, training_labels)

    # -------------------------------------------------------------------------
    # TEST LEARNED SVM MODEL ON TEST DATA.
    # REPORT ACCURACY, PRECISION, AND RECALL
    # (USING THRESHOLD 0 TO DETERMINE POSITIVE AND NEGATIVE CLASSIFICATIONS)
    # -------------------------------------------------------------------------
    print 'Testing the trained classifier with the best C value..'

    decisions = clf.decision_function(testing_features)
    tp, fn, fp, tn, = 0, 0, 0, 0
    for i in range(len(decisions)):
        # count tp, fn, fp, tn
        label = testing_labels[i]
        decision = decisions[i]
        if label == 1:
            if decision >= 0:
                tp += 1
            elif decision < 0:
                fn += 1
        elif label == 0:
            if decision >= 0:
                fp += 1
            elif decision < 0:
                tn += 1
    accuracy = float(tp + tn) / (tp + tn + fp + fn)
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)  # True positive rate
    print 'Using threshold = 0:\nAccuracy: %f, Precision: %f, Recall: %f\n' % (
    accuracy, precision, recall)

    # -------------------------------------------------------------------------
    # USE RESULTS ON TEST DATA TO CREATE AN ROC CURVE FOR THIS SVM, 
    # USING ABOUT 200 EVENLY SPACED THRESHOLDS.
    # -------------------------------------------------------------------------
    # Jon Shephardson: Use svm.decision(X,y) to produce the distance from
    # sv's for every test.
    # Then find the min and Max in these to determine the range of classifier.
    # Take that range and chop it up into 200 equal thresholds. Now calculate
    # your tpr and fpr for each threshold. Graph it.
    # -------------------------------------------------------------------------
    print 'Creating an ROC curve for the trained classifier on the test set..\n'
    range_of_classifier = decisions.max() - decisions.min()
    threshold_increment = float(range_of_classifier) / 200
    starting_threshold = decisions.min()
    current_threshold = starting_threshold

    ROCinfo = list()
    number_of_increments = 0
    while number_of_increments < 200:
        # Decisions don't change, but if the score >= current_threshold, 
        # say the class is 1 and compare to the label.
        tp, fn, fp, tn, = 0, 0, 0, 0

        for i in range(len(decisions)):
            # count tp, fn, fp, tn
            label = testing_labels[i]
            decision = decisions[i]
            if label == 1:
                if decision >= current_threshold:
                    tp += 1
                elif decision < current_threshold:
                    fn += 1
            elif label == 0:
                if decision >= current_threshold:
                    fp += 1
                elif decision < current_threshold:
                    tn += 1
        true_positive_rate = float(tp) / (tp + fn)
        false_positive_rate = float(fp) / (fp + tn)

        ROCinfo.append([false_positive_rate, true_positive_rate])

        # current_threshold = current_threshold + threshold_increment
        current_threshold += threshold_increment
        number_of_increments += 1

    # WRITE OUT THE ROC CURVE DATA
    filename = "experiment1ROC.csv"
    print 'Saving ROC data to %s\n' % (filename)
    f = open(filename, 'w')
    for row in ROCinfo:
        f.write(str(row[0]))
        f.write(',')
        f.write(str(row[1]))
        f.write('\n')
    f.close()

    # MAKES A PLOT THAT LOOKS NICER THAN MY METHOD.
    # fpr, tpr, thresholds = roc_curve(testing_labels, decisions, pos_label=1)
    # roc_auc = auc(fpr, tpr)
    # plt.title('ROC Curve Experiment 1')
    # plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0,1],[0,1],'r--')
    # plt.xlim([-0.1,1.2])
    # plt.ylim([-0.1,1.2])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()

    print "\nEnd of experiment 1..\n"
    # =========================================================================




    # =========================================================================
    print "\nBeginning experiment 2..\n"

    # GET WEIGHT VECTOR FROM THE CLASSIFIER
    weight_vector = clf.coef_[0]
    # The most significant weights indices will be collected from this vector
    absolute_weight_vector = np.array(
        [np.abs(weight) for weight in weight_vector])

    accuracies_from_feature_selection = list()
    # Select features:
    # For m = 2 to 57
    for m in range(2, 57 + 1):
        print 'Selecting the largest %d weights from the classifier\'s weight ' \
              'vector..' % (m)

        # Selecting the indices of the m largest weights (by absolute value)
        indices = (-absolute_weight_vector).argsort()[:m]
        m_training_features = FeatureSubset(training_features, indices)
        m_testing_features = FeatureSubset(testing_features, indices)

        # Train a linear SVM, SVMm , on all the training data,
        # only using these m features, and using C* from Experiment 1
        SVMm = svm.SVC(kernel='linear', C=bestC)
        SVMm.fit(m_training_features, training_labels)

        # Test SVMm on the test set to obtain accuracy.
        predictions = SVMm.predict(m_testing_features)
        tp, tn, fp, fn = 0, 0, 0, 0
        correct_count_j = 0
        for index in range(len(predictions)):
            prediction = predictions[index]
            actual = testing_labels[index]
            if prediction == actual:  # make sure types are the same
                correct_count_j += 1
                if actual == 0:
                    tn += 1
                elif actual == 1:
                    tp += 1
            else:
                if prediction == 1 and actual == 0:
                    fp += 1
                elif prediction == 0 and actual == 1:
                    fn += 1
        acc = float(tp + tn) / (tp + tn + fp + fn)
        accuracies_from_feature_selection.append([m, acc])

    # Plot accuracy vs. m
    filename2 = "experiment2AccuracyMFeatures.csv"
    print 'Saving ROC data to %s\n' % (filename2)
    f = open(filename2, 'w')
    f.write('m,accuracy')
    f.write('\n')
    for row in accuracies_from_feature_selection:
        f.write(str(row[0]))
        f.write(',')
        f.write(str(row[1]))
        f.write('\n')
    f.close()

    print "\nEnd of experiment 2..\n"
    # =========================================================================


    # =========================================================================
    print "\nBeginning experiment 3..\n"
    # Same as Experiment 2, but for each m, select m features at random from
    # the complete set.
    # This is to see if using SVM weights for feature selection has any
    # advantage over random.

    accuracies_from_random_feature_selection = list()
    # Select features:
    # For m = 2 to 57
    for m in range(2, 57 + 1):
        print 'Selecting %d random weights from the classifier\'s weight ' \
              'vector..' % (
        m)

        # Random indices
        indices = [np.random.randint(2, 57) for i in range(m)]
        m_training_features = FeatureSubset(training_features, indices)
        m_testing_features = FeatureSubset(testing_features, indices)

        # Train a linear SVM, SVMm , on all the training data,
        # only using these m features, and using C* from Experiment 1
        SVMm = svm.SVC(kernel='linear', C=bestC)
        SVMm.fit(m_training_features, training_labels)

        # Test SVMm on the test set to obtain accuracy.
        predictions = SVMm.predict(m_testing_features)
        tp, tn, fp, fn = 0, 0, 0, 0
        correct_count_j = 0
        for index in range(len(predictions)):
            prediction = predictions[index]
            actual = testing_labels[index]
            if prediction == actual:  # make sure types are the same
                correct_count_j += 1
                if actual == 0:
                    tn += 1
                elif actual == 1:
                    tp += 1
            else:
                if prediction == 1 and actual == 0:
                    fp += 1
                elif prediction == 0 and actual == 1:
                    fn += 1
        acc = float(tp + tn) / (tp + tn + fp + fn)
        accuracies_from_random_feature_selection.append([m, acc])

    # Plot accuracy vs. m
    filename3 = "experiment3AccuracyMRandomFeatures.csv"
    print 'Saving ROC data to %s\n' % (filename3)
    f = open(filename3, 'w')
    f.write('m,accuracy')
    f.write('\n')
    for row in accuracies_from_feature_selection:
        f.write(str(row[0]))
        f.write(',')
        f.write(str(row[1]))
        f.write('\n')
    f.close()

    print "\nEnd of experiment 3..\n"
    # =========================================================================


if __name__ == "__main__":
    main()
