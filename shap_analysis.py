# import shap
import shap
import argparse
import tensorflow as tf
import os
from preprocess import get_data

script_dir = os.path.dirname(__file__)
weights_dir = os.path.join(script_dir,"model_checkpts")

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_weights", type=str, default="deadbeef") 
    args = parser.parse_args()
    return args

def main(args):
    if args.load_weights != "deadbeef":
        model_load_name = args.load_weights
        model = tf.keras.models.load_model(os.path.join(weights_dir, model_load_name))
        print("model loaded")
    else:
        print("Must specify which model to load")
        raise Exception

    script_dir = os.path.dirname(__file__)
    train_path = os.path.join(script_dir, "data/train.csv")
    test_path = os.path.join(script_dir, "data/test.csv")
    train_abstracts, train_labels, test_abstracts, test_labels = get_data(train_path, test_path)
    print("data loaded")

    print("shap dartasets", type(shap.datasets.diabetes()))
    print(shap.datasets.diabetes()[:3])
    print(train_abstracts[:3])
    explainer = shap.DeepExplainer(model, train_abstracts[:100])
    print("created explainer")
    shap_values = explainer(test_abstracts[:10])
    print("got shap values")
    shap.plots.waterfall(shap_values[0], max_display=20, show=True)
    print("visualized")


if __name__ == "__main__":
    args = parseArguments()
    main(args)


