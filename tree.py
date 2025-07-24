import numpy as np
from sklearn.tree import DecisionTreeClassifier

import stats
from stats import Stats, Selection
from typing import Dict, Tuple, List
from collections import defaultdict
import sys
import warnings

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
            class_weight='balanced',
            criterion='entropy'  # 'gini'
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
                if move_count < lowest:
                    lowest = move_count
                    best = solutions[i]
                if move_count > highest:
                    highest = move_count
                    worst = solutions[i]

            node = tree_viz.TreeNode(
                text=f"{condition}\n{100 * fraction_of_cases:0.1f}% of cases ({n_samples}/{len(y)})\n{metric}={100 * positive_fraction:0.1f}%\nbest: {best} ({lowest})\nworst: {worst} ({highest})",
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


def load(filename: str = "full_data.csv") -> Tuple[
    Dict[Tuple[str, int], List[Tuple[List, bool]]], List[str]]:
    data_by_drm = defaultdict(list)
    solutions = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            bucket, sol = Stats.parse_line(line)
            features = [
                bucket.corner_arm,
                bucket.edge_arm,
                bucket.corner_orbit_split,
                bucket.corner_orbit_parity,
                bucket.n_fake_pairs,
                bucket.n_side_pairs,
                bucket.n_pairs,
                bucket.corner_arm_split,
                bucket.edge_arm_split,
                # bucket.n_fake_pairs,
                # bucket.n_side_pairs,
            ]
            corner_case = stats.corner_case_name(bucket.n_bad_corners, bucket.corner_orbit_split,
                                                 bucket.corner_orbit_parity)
            data_by_drm[corner_case].append((features, bucket.move_count < 7))
            solutions.append(sol)
    return data_by_drm, solutions


def main(drm_c, drm_e):
    data, solutions = load(f"{drm_c}c{drm_e}e.csv")
    for corner_case, xy in data.items():
        title=f"{corner_case}{drm_e}e"
        print(f"Training decision tree for {title}")
        feature_names = [
            "n_pairs",
            "corner_arm",
            "edge_arm",
            "corner_orbit_split",
            "corner_orbit_parity",
            "n_fake_pairs",
            "n_side_pairs",
        ]
        y = np.array([1 if l else 0 for _, l in xy])
        X = np.array([np.array(x) for x, _ in xy])

        # Initialize and train the model
        trainer = TargetMetricDecisionTree(max_depth=6)

        # Train to target metrics
        trainer.train_to_target(X, y, feature_names)
        metric = "p_sub7"
        viz = trainer.visualize(metric, X, y, solutions)
        viz.save_and_open(title, filename=f"drm_trees/{title}_{metric}.html")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        drm_c = int(sys.argv[1])
        drm_e = int(sys.argv[2])
    else:
        print("Usage: python tree.py [corners] [edges]")
        sys.exit(1)
    main(drm_c, drm_e)
