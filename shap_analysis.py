# import shap
import shap
import argparse
import tensorflow as tf
import os
from preprocess import get_data
from model import CombinedModel

script_dir = os.path.dirname(__file__)
weights_dir = os.path.join(script_dir,"model_checkpts")

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_weights", type=str, default="deadbeef") 
    parser.add_argument("--bert", action="store_true") 
    parser.add_argument("--max_num_tokens", type=int, default=512)
    args = parser.parse_args()
    return args

def main(args):
    tf.compat.v1.enable_eager_execution()
    if args.load_weights != "deadbeef":
        model_load_name = args.load_weights
        loaded_model = CombinedModel(model_load_name, args)
        print("model loaded")
    else:
        print("Must specify which model to load")
        raise Exception

    script_dir = os.path.dirname(__file__)
    train_path = os.path.join(script_dir, "data/train.csv")
    test_path = os.path.join(script_dir, "data/test.csv")
    train_abstracts, train_labels, test_abstracts, test_labels = get_data(train_path, test_path)
    print("data loaded")

    inputs = tf.keras.Input(shape=train_abstracts[:100].shape)
    outputs = loaded_model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.set_weights(loaded_model.get_weights())
    explainer = shap.DeepExplainer(model, train_abstracts[:100])
    print("created explainer")
    shap_values = explainer(test_abstracts[:10])
    print("got shap values")
    shap.plots.waterfall(shap_values[0], max_display=20, show=True)
    print("visualized")


if __name__ == "__main__":
    args = parseArguments()
    main(args)


