import numpy as np
from sklearn.tree import DecisionTreeClassifier

import stats
from stats import Stats, Selection
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import sys
import warnings
import math

warnings.filterwarnings('ignore')
import tree_viz


class TargetMetricDecisionTree:
    def __init__(self, max_depth=20):
        """
        Initialize the decision tree trainer with target metrics.

        Args:
            max_depth: Maximum depth to search
        """
        self.max_depth = max_depth
        self.model = None

    def train_to_target(self, X_train, y_train, feature_names=None):
        """
        Train decision tree to minimum depth that achieves target metrics.

        Args:
            X_train, y_train: Training data
            feature_names: List of feature names for visualization
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        self.feature_names = feature_names

        # Train model with current depth
        model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=42,
            class_weight='balanced', # {0: 0.1, 1: 0.9},
            criterion='entropy',  # 'gini',
            min_impurity_decrease=0.01
        )
        model.fit(X_train, y_train)
        self.model = model

    def get_node_metrics(self, node_samples, node_values, y_val):
        """Calculate precision and recall for a specific node's samples."""
        if len(node_samples) == 0:
            return 0.0, 0.0, 0

        # Get predictions for samples that reach this node
        node_labels = y_val[node_samples]
        positive_samples = np.sum(node_labels == 1)
        total_samples = len(node_labels)

        if total_samples == 0:
            return 0.0, 0.0, 0

        # For leaf nodes, calculate actual precision/recall
        if positive_samples == 0:
            precision = 0.0
            recall = 0.0
        else:
            # Precision: of samples in this node, what fraction are actually positive
            precision = positive_samples / total_samples
            # Recall: of all positive samples, what fraction reach this node
            total_positives = np.sum(y_val == 1)
            recall = positive_samples / total_positives if total_positives > 0 else 0.0

        return precision, recall, total_samples

    def get_samples_for_node(self, X, node_id):
        """Get the indices of validation samples that reach a specific node."""
        # Get the decision path for all validation samples
        decision_paths = self.model.decision_path(X)

        # Find samples that pass through this node
        node_samples = []
        for i in range(len(X)):
            if decision_paths[i, node_id] == 1:  # Sample passes through this node
                node_samples.append(i)

        return np.array(node_samples)

    def visualize(self, metric, X, y, solutions: List[str]):
        tree_structure = self.model.tree_
        feature_names = self.feature_names

        def build_node(node_id, condition: str) -> tree_viz.TreeNode:
            feature = tree_structure.feature[node_id]
            threshold = tree_structure.threshold[node_id]

            # Get samples that reach this node and calculate detailed metrics
            node_sample_indices = self.get_samples_for_node(X, node_id)
            n_samples = len(node_sample_indices)
            positive_samples = np.sum(y[node_sample_indices] == 1)
            fraction_of_cases = n_samples / len(y)
            positive_fraction = positive_samples / len(node_sample_indices)
            mutual_info = -math.log(1 - positive_fraction * positive_fraction * fraction_of_cases)
            color = "#ffffff"
            if positive_fraction >= 0.5:
                color = "#00dd00"
            elif positive_fraction > 0.33:
                color = "#88ff88"
            elif positive_fraction >= .20:
                color = "#aaffaa"
            elif positive_fraction >= .10:
                color = "#ddffdd"

            best, lowest = "", 99
            worst, highest = "", 0
            for i in node_sample_indices:
                move_count = len(solutions[i].split(" "))
                if y[i] and move_count < lowest:
                    lowest = move_count
                    best = solutions[i]
                if move_count > highest:
                    highest = move_count
                    worst = solutions[i]

            node = tree_viz.TreeNode(
                text=f"{condition}\n{100 * fraction_of_cases:0.1f}% of cases ({n_samples}/{len(y)})\n{metric}={100 * positive_fraction:0.1f}%\nmutual info: {mutual_info:.1e}\nbest: {best} ({lowest})\nworst: {worst} ({highest})",
                style={"color": color}
            )

            # If not a leaf node
            if feature >= 0:
                feature_name = feature_names[feature]
                th = int(threshold)

                # Left child (condition is True)
                left_child = tree_structure.children_left[node_id]
                node.left = build_node(left_child,
                                       f"{feature_name} {'=' if th == 0 else '<='} {th}")

                # Right child (condition is False)
                right_child = tree_structure.children_right[node_id]
                node.right = build_node(right_child, f"{feature_name} >= {th + 1}")

            return node

        root = build_node(0, "All")
        return tree_viz.BinaryTreeHTMLGenerator(root)


def load(drm_c, drm_e, target_move_count, difficulty: str, split_subcases: bool) -> Tuple[Dict[Tuple[str, int], List[Tuple[List, bool]]], Dict[str, List[str]]]:
    if difficulty is not None and difficulty not in ["easy", "findable"]:
        raise f"Unknown difficulty {difficulty}"
    filename=f"{drm_c}c{drm_e}e.csv"
    data_by_drm = defaultdict(list)
    solutions = defaultdict(list)
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            bucket, sol = Stats.parse_line(line)
            features = [
                bucket.corner_arm,
                bucket.edge_arm,
                bucket.n_pairs,
                bucket.n_fake_pairs,
                bucket.n_side_pairs,
            ]
            corner_case = f"{drm_c}c"
            if split_subcases:
                corner_case = stats.corner_case_name(bucket)
            label = bucket.move_count <= target_move_count
            if difficulty == "easy" and bucket.difficulty > 0:
                label = False
            if difficulty == "findable" and bucket.difficulty > 1:
                label = False
            data_by_drm[corner_case].append((features, label))
            solutions[corner_case].append(sol)
    return data_by_drm, solutions


def main(drm_c, drm_e, target_move_count, difficulty: str, split_subcases: bool):
    data, solutions = load(drm_c, drm_e, target_move_count, difficulty, split_subcases)
    for corner_case, xy in data.items():
        title=f"{corner_case}{drm_e}e"
        print(f"Training decision tree for {title}")
        feature_names = [
            "corner_arm",
            "edge_arm",
            "n_pairs",
            "n_fake_pairs",
            "n_side_pairs",
        ]
        y = np.array([1 if l else 0 for _, l in xy])
        X = np.array([np.array(x) for x, _ in xy])

        # Initialize and train the model
        trainer = TargetMetricDecisionTree(max_depth=6)

        # Train to target metrics
        trainer.train_to_target(X, y, feature_names)
        metric = f"p_sub{target_move_count+1}"
        if difficulty is not None:
            metric = f"{metric}_{difficulty}"
        viz = trainer.visualize(metric, X, y, solutions[corner_case])
        filename=f"drm_trees/{title}_{metric}.html"
        viz.save_and_open(title, filename)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        drm_c = int(sys.argv[1])
        drm_e = int(sys.argv[2])
        target_move_count = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        difficulty = sys.argv[4] if len(sys.argv) > 4 else None
        split_subcases = sys.argv[5].lower() == "true" if len(sys.argv) > 5 else False
    else:
        print("Usage: python tree.py [corners] [edges]")
        sys.exit(1)
    main(drm_c, drm_e, target_move_count, difficulty, split_subcases)
