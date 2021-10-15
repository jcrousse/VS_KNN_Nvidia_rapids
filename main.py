import argparse
import datetime
import numpy as np
from vs_knn import VsKnnModel
from vs_knn.train_test_split import train_test_split


def get_arguments():
    """Get this script's command line arguments"""
    parser = argparse.ArgumentParser(description='cuDF implementation of VS-KNN')
    parser.add_argument('--train', '-t', dest='train', action='store_true', help="train the model")
    parser.set_defaults(train=False)
    parser.add_argument('--split', '-s', dest='split', action='store_true', help="split the dataset into train/test")
    parser.set_defaults(split=False)
    parser.add_argument('--predict', '-p', dest='predict', action='store_true', help="predict on test dataset")
    parser.set_defaults(predict=False)
    parser.add_argument('--no-cudf', '-c', dest='no_cudf', action='store_true', help="use pandas instead of cudf")
    parser.set_defaults(no_cudf=False)
    args = parser.parse_args()
    return args.train, args.split, args.predict, args.no_cudf


if __name__ == '__main__':
    train, split, predict, no_cudf = get_arguments()

    model = VsKnnModel('config.json')

    if split:
        train_test_split()
    if train:
        model.train()
    else:
        model.load()
    if predict:
        test_examples = model.get_test_dict()

        examples_n = list(test_examples.keys())[0:100]
        time_per_prediction = np.zeros(len(examples_n))

        for idx, user_session in enumerate(examples_n):
            query_items = test_examples[user_session]

            start = datetime.datetime.now()
            items_scores = model.predict(query_items)
            end = datetime.datetime.now()
            delta = end - start
            time_per_prediction[idx] = int(delta.total_seconds() * 1000)

        print("average duration: ", np.average(time_per_prediction), " milliseconds")
        print("p90 duration: ", np.percentile(time_per_prediction, 90), " milliseconds")


# Todo:
#     -wget dataset if not present
#     -Compare perfs pandas - cudf ?
#     -Box plots of time spent on each step ?




