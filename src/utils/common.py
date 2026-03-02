"""Common utilities and helper functions"""

import os
import numpy as np


def ensure_directory(path: str) -> str:
    """
    Ensure directory exists, create if needed.
    
    Args:
        path: Directory path
    
    Returns:
        str: The directory path
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_project_root() -> str:
    """
    Get the project root directory.
    
    Returns:
        str: Absolute path to project root
    """
    return os.path.dirname(os.path.abspath(__file__))


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import tensorflow as tf
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class DatasetStatistics:
    """Calculate and store dataset statistics"""
    
    @staticmethod
    def class_distribution(Z: np.ndarray, class_names: list) -> dict:
        """
        Calculate class distribution.
        
        Args:
            Z: Target labels array
            class_names: List of class names
        
        Returns:
            dict: Distribution of each class
        """
        distribution = {}
        for i, name in enumerate(class_names):
            count = np.sum(Z[:, i])
            percentage = (count / len(Z)) * 100
            distribution[name] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }
        return distribution
    
    @staticmethod
    def print_distribution(Z: np.ndarray, class_names: list):
        """Print class distribution nicely"""
        dist = DatasetStatistics.class_distribution(Z, class_names)
        print("\nClass Distribution:")
        print("-" * 50)
        for class_name, stats in dist.items():
            print(f"{class_name:10} | Count: {stats['count']:5} | "
                  f"Percentage: {stats['percentage']:6.2f}%")
        print("-" * 50)
