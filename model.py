from preprocess import get_data
from transformers import AutoTokenizer, TFDistilBertModel, TFBertModel
import tensorflow as tf
import numpy as np
import math
import argparse
import os

track_checkpoint_fname = 'track_checkpoint.txt'
weights_dir = "model_checkpts"

def parseArguments():
    parser = argparse.ArgumentParser()
    ### ONLY USE ONE OF (--load_weights, --save_weights, --test_gui) ###
    parser.add_argument("--load_weights", action="store_true") 
    # only if continuing to train on existing weights - if just testing use --test_only
    # also this will automatically save the weights back as well
    parser.add_argument("--save_weights", action="store_true")
    parser.add_argument("--bert", action="store_true")
    # can't select both --save_weights and --test_only, it'll just train and save the weights
    # only select this if you want to reset training and not continue from the last checkpoint
    parser.add_argument("--test_gui", action="store_true")
    # if this is selected, will automatically load weights and start single-test interface
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=10)
    # will probably need more than 10 epochs in final training
    parser.add_argument("--max_num_tokens", type=int, default=512)
    # maximum number of tokens considered in each input abstract (for both training & testing)
    args = parser.parse_args()
    return args

class GPTClassifier(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.seq_model = tf.keras.Sequential([
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(50),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(2),
            tf.keras.layers.Activation('sigmoid')
        ])

    def call(self, inputs):
        outputs = self.seq_model(inputs)
        return outputs

def train(model, train_abstracts, train_labels, args):
    batch_size = args.batch_size
    num_batches = math.floor(train_abstracts.shape[0] / batch_size)
    indices = tf.random.shuffle(tf.range(train_abstracts.shape[0]-1))
    train_abstracts = tf.gather(train_abstracts, indices)
    train_labels = tf.gather(train_labels, indices)
    train_abstracts = train_abstracts[:batch_size*num_batches]
    train_labels = train_labels[:batch_size*num_batches]
    if args.bert:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    else:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        bert_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    # train_abstracts_list = train_abstracts.numpy().tolist()
    # train_abstracts_list = [s.decode('utf-8') for s in train_abstracts_list]
    # tokenized_abstracts = tokenizer(train_abstracts_list, return_tensors='tf', max_length=512, padding='max_length', truncation=True)
    # print("tokenized abstracts", type(tokenized_abstracts))
    total_loss = 0
    for i in range(num_batches):
        batch_abstracts = train_abstracts[i * batch_size:(i+1)*batch_size]
        batch_labels = train_labels[i * batch_size:(i+1)*batch_size]
        batch_abstracts_list = batch_abstracts.numpy().tolist()
        batch_abstracts_list = [s.decode('utf-8') for s in batch_abstracts_list]
        tokenized_abstracts = tokenizer(batch_abstracts_list, return_tensors='tf', max_length=args.max_num_tokens, \
                                        padding='max_length', truncation=True) # default max_length 512
        hidden_states = bert_model(tokenized_abstracts).last_hidden_state
        hidden_states = tf.reshape(hidden_states, (batch_size, -1))

        with tf.GradientTape() as tape:
            outputs = model(hidden_states)
            loss = model.loss(outputs, batch_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss
    return total_loss / num_batches

## NOTE: should we defined get_config and from_config functions to specify saving/loading weights?
# for that matter, do we need to save the compilatoin information & optimizer state? it's helpful
# if we are continuing to train the model, but not otherwise

def test(model, test_abstracts, test_labels, args):
    print("testing")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    distil_bert = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    test_abstracts_list = test_abstracts.numpy().tolist()
    test_abstracts_list = [s.decode('utf-8') for s in test_abstracts_list]
    tokenized_abstracts = tokenizer(test_abstracts_list, return_tensors='tf', max_length=args.max_num_tokens, \
                                    padding='max_length', truncation=True) # default max_length 512
    hidden_states = distil_bert(tokenized_abstracts).last_hidden_state
    hidden_states = tf.reshape(hidden_states, (hidden_states.shape[0], -1))
    outputs = model(hidden_states)

    metric = tf.keras.metrics.CategoricalAccuracy()
    metric.update_state(test_labels, outputs)
    return metric.result().numpy() # return accuracy on training data

def test_one(model, test_abstract, args):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    distil_bert = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    # don't need to decode abstract as utf-8 because if it's input from the user it shouldn't be a bytestring
    tokenized_abstract = tokenizer([test_abstract], return_tensors='tf', max_length=args.max_num_tokens, \
                                   padding='max_length', truncation=True) # default max_length 512
    hidden_states = distil_bert(tokenized_abstract).last_hidden_state
    hidden_states = tf.reshape(hidden_states, (hidden_states.shape[0], -1))
    output = model(hidden_states)
    return output # index that should have the higher value: 0 if human, 1 if chatgpt

def main(args):
    # load data
    train_abstract, train_labels, test_abstracts, test_labels = get_data('data/train.csv', 'data/test.csv')
    print("Data loaded")

    # read what number the last training run was (this is unnecessary if we're not testing or saving/loading weights)
    with open(track_checkpoint_fname, 'r') as f:
        last_checkpoint = int(f.read().strip())
    last_checkpoint_fname = f"model{last_checkpoint}"

    if not args.test_gui:
        # load or create new model
        if args.load_weights:
            print("path: ", os.path.join(weights_dir, last_checkpoint_fname))
            model = tf.keras.models.load_model(os.path.join(weights_dir, last_checkpoint_fname))
        else:
            model = GPTClassifier()
        # *** train model ***
        for i in range(args.num_epochs):
            total_loss = train(model, train_abstract, train_labels, args)
            print("Epoch", i, ": total_loss = ", total_loss)
        # save model if necessary
        if args.save_weights or args.load_weights:
            # update the number of the latest training checkpoint
            with open(track_checkpoint_fname, 'w') as f:
                f.write(str(last_checkpoint + 1))
            model.save(os.path.join(weights_dir, f"model{last_checkpoint + 1}"))
            print("Saved new weights")
        # test model
        accuracy = test(model, test_abstracts, test_labels, args)
        print("Testing accuracy: ", accuracy)
    else:
        # run simple gui interface so you can try it yourself!
        # TODO: make the interface nice with tkinter
        test_abstract = input("Paste an abstract here: ").strip()
        model = tf.keras.models.load_model(os.path.join(weights_dir, last_checkpoint_fname))
        guess = test_one(model, test_abstract, args)
        print(f"This is probably written by {np.array(['a human', 'chatgpt'])[tf.math.argmax(guess[0])]} with probabilities [human, chatgpt] = {tf.nn.softmax(guess[0])}")

if __name__ == "__main__":
    args = parseArguments()
    main(args)