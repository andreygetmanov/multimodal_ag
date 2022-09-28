import pandas as pd
import os

from autogluon.tabular import TabularPredictor


def run_autogluon_example(file_path: str, time_limit: int = 10) -> float:
    """
    This is an example of AutoGluon use on multimodal data.
    The data is taken from Jigsaw 2019 Kaggle competition:
    https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
    and contains information about the post (e.g. likes, rating, date created, etc.).
    Column that contains text features is 'created_date'.
    Other columns contain numerical and categorical features.
    The task is to predict whether online social media comments are toxic
    based on their text and additional tabular features.

    :param file_path: path to the file with multimodal data
    :param time_limit: time limit in seconds for AutoGluon training

    :return: F1 metrics of the model
    """
    root = os.path.abspath(file_path)
    train_path = os.path.join(root, 'jigsaw_unintended_bias100K_train.csv')
    test_path = os.path.join(root, 'jigsaw_unintended_bias100K_test.csv')
    fit_data = pd.read_csv(train_path)
    predict_data = pd.read_csv(test_path)

    predictor = TabularPredictor(label='target', eval_metric='f1', path='jigsaw_model')
    predictor.fit(fit_data, hyperparameters='multimodal', presets='best_quality', time_limit=time_limit * 60)

    metrics = predictor.evaluate(predict_data)
    f1 = round(metrics.get('f1'), 3)
    print(f'F1 for validation sample is {f1}')

    return f1


if __name__ == '__main__':
    run_autogluon_example(file_path='jigsaw_unintended_bias100K', time_limit=10)