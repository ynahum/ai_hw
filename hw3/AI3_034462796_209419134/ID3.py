import math

from DecisonTree import Leaf, Question, DecisionNode, class_counts
from utils import *

"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, label_names: list, min_for_pruning=0, target_attribute='diagnosis'):
        self.label_names = label_names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()
        self.min_for_pruning = min_for_pruning

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """
        # TODO:
        #  Calculate the entropy of the data as shown in the class.
        #  - You can use counts as a helper dictionary of label -> count, or implement something else.

        counts = class_counts(rows, labels)
        impurity = 0.0

        # ====== YOUR CODE: ======
        num_of_rows = rows.shape[0]
        for k, v in counts.items():
            prob = v/num_of_rows
            if prob > 0:
                impurity -= prob * np.log2(prob)
        # ========================

        return impurity

    def info_gain(self, left, left_labels, right, right_labels, current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        # TODO:
        #  - Calculate the entropy of the data of the left and the right child.
        #  - Calculate the info gain as shown in class.
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        info_gain_value = 0.0
        # ====== YOUR CODE: ======
        entropy_left = ID3.entropy(left, left_labels)
        entropy_right = ID3.entropy(right, right_labels)
        num_of_left_samples = float(left.shape[0])
        num_of_right_samples = float(right.shape[0])
        num_of_samples = num_of_left_samples + num_of_right_samples
        weighted_entropy = (num_of_left_samples * entropy_left + num_of_right_samples * entropy_right) / num_of_samples
        info_gain_value = current_uncertainty - weighted_entropy
        # ========================

        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """
        # TODO:
        #   - For each row in the dataset, check if it matches the question.
        #   - If so, add it to 'true rows', otherwise, add it to 'false rows'.
        #   - Calculate the info gain using the `info_gain` method.

        gain, true_rows, true_labels, false_rows, false_labels = None, None, None, None, None
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'

        # ====== YOUR CODE: ======
        num_of_cols = rows.shape[1]
        true_rows = np.empty((0,num_of_cols))
        true_labels = np.empty((0,1))
        false_rows = np.empty((0,num_of_cols))
        false_labels = np.empty((0,1))
        for idx, row in enumerate(rows):
            if question.match(row):
                true_rows = np.vstack([true_rows, row])
                true_labels = np.vstack([true_labels, labels[idx]])
            else:
                false_rows = np.vstack([false_rows, row])
                false_labels = np.vstack([false_labels, labels[idx]])

        true_labels = np.squeeze(true_labels, axis=(1,))
        false_labels = np.squeeze(false_labels, axis=(1,))
        num_of_true_samples = len(true_labels)
        num_of_false_samples = len(false_labels)
        if num_of_true_samples == 0 or num_of_false_samples == 0:
            print(f"found num_of_true_samples == 0 or num_of_false_samples == 0")
            gain = 0
        else:
            gain = self.info_gain(true_rows, true_labels, false_rows, false_labels, current_uncertainty)
        # ========================

        return gain, true_rows, true_labels, false_rows, false_labels

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        # TODO:
        #   - For each feature of the dataset, build a proper question to partition the dataset using this feature.
        #   - find the best feature to split the data. (using the `partition` method)
        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)

        # ====== YOUR CODE: ======

        assert (len(set(labels)) > 1)

        num_of_cols = rows.shape[1]

        # iterate over all features
        for col_idx in range(num_of_cols):
            if col_idx in self.used_features:
                continue

            # assuming all features are continuous we look for the best value to split with
            col = rows[:,col_idx]
            col_sort_indexes = col.argsort()
            col_sorted = col[col_sort_indexes]
            labels_sorted = labels[col_sort_indexes]

            split_values = []

            # find first index from which all labels are the same from the start
            first_split_index = 0
            first_label = labels_sorted[first_split_index]
            for idx, label in enumerate(labels_sorted):
                if label == first_label:
                    continue
                else:
                    first_split_index = idx - 1
                    break

            # find last index from which all labels are the same till the end
            last_split_index = len(labels_sorted) - 1
            last_label = labels_sorted[last_split_index]
            for idx, label in reversed(list(enumerate(labels_sorted))):
                if label == last_label:
                    continue
                else:
                    last_split_index = idx
                    break

            assert(first_split_index <= last_split_index)

            # calculate possible split values
            for idx, _ in enumerate(col_sorted):
                if idx < first_split_index or idx > last_split_index:
                    continue
                else:
                    current_val = (col_sorted[idx] + col_sorted[idx + 1])/2
                    # another optimization not to take the same values in consideration
                    if len(split_values) > 0 and split_values[-1] == current_val:
                        continue
                    split_values.append(current_val)

            # iterate over all possible split values for the best
            for idx, value in enumerate(split_values):
                question = Question(col, col_idx, value)
                gain, true_rows, true_labels, false_rows, false_labels =\
                    self.partition(rows, labels, question, current_uncertainty)
                # greater or EQUAL to take the last feature that has the best IG as described
                if gain >= best_gain:
                    best_gain = gain
                    best_question = question
                    best_false_rows = false_rows
                    best_false_labels = false_labels
                    best_true_rows = true_rows
                    best_true_labels = true_labels
        # ========================

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer.
                or leaf if we have to prune this branch (in which cases ?)

        """
        # TODO:
        #   - Try partitioning the dataset using the feature that produces the highest gain.
        #   - Recursively build the true, false branches.
        #   - Build the Question node which contains the best question with true_branch, false_branch as children
        best_question = None
        true_branch, false_branch = None, None

        # ====== YOUR CODE: ======
        # done in leaf build already
        #counts = class_counts(rows, labels)
        #majority_class = max(counts, key=counts.get)

        # we assume that since this is a binary tree, we don't have to check
        # for an empty child
        num_of_different_labels = len(set(labels))
        assert(num_of_different_labels > 0)

        # stopping conditions
        if (num_of_different_labels == 1) or (self.min_for_pruning > 0 and rows.shape[0] < self.min_for_pruning):
            return Leaf(rows, labels)

        # we cannot run out of features as all are continuous and we don't put it aside
        _, best_question, t_rows, t_labels, f_rows, f_labels = self.find_best_split(rows, labels)

        # inner function for checking if child is a leaf due to pruning.
        # if yes, we set it as a Leaf with parent (current) node rows and labels
        def is_branch_child_pruned(branch, branch_rows, branch_labels, min_for_pruning):
            return min_for_pruning > 0 and isinstance(branch, Leaf) and\
                   branch_rows.shape[0] < min_for_pruning and len(set(branch_labels)) > 1

        true_branch = self.build_tree(t_rows, t_labels)
        if is_branch_child_pruned(true_branch, t_rows, t_labels, self.min_for_pruning):
            true_branch = Leaf(rows, labels)

        false_branch = self.build_tree(f_rows, f_labels)
        if is_branch_child_pruned(false_branch, f_rows, f_labels, self.min_for_pruning):
            false_branch = Leaf(rows, labels)
        # ========================

        return DecisionNode(best_question, true_branch, false_branch)

    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        # TODO: Build the tree that fits the input data and save the root to self.tree_root

        # ====== YOUR CODE: ======
        self.tree_root = self.build_tree(x_train, y_train)
        # ========================

    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """
        # TODO: Implement ID3 class prediction for set of data.
        #   - Decide whether to follow the true-branch or the false-branch.
        #   - Compare the feature / value stored in the node, to the example we're considering.

        if node is None:
            node = self.tree_root
        prediction = None

        # ====== YOUR CODE: ======
        if isinstance(node,Leaf):
            prediction = max(node.predictions, key=node.predictions.get)
        else:
            question = node.question
            if question.match(row):
                prediction = self.predict_sample(row, node.true_branch)
            else:
                prediction = self.predict_sample(row, node.false_branch)
        # ========================

        return prediction

    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """
        # TODO:
        #  Implement ID3 class prediction for set of data.

        y_pred = None

        # ====== YOUR CODE: ======
        y_pred = np.empty((0,1))
        for idx in range(rows.shape[0]):
            pred = self.predict_sample(rows[idx], self.tree_root)
            y_pred = np.vstack([y_pred, pred])
        y_pred = np.squeeze(y_pred, axis=(1,))
        # ========================

        return y_pred
