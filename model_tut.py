# This code was written by following Ray William's tutorial on finetuning distilBERT for binary classification:
# https://towardsdatascience.com/hugging-face-transformers-fine-tuning-distilbert-for-binary-classification-tasks-490f1d192379

from preprocess import get_data
from transformers import AutoTokenizer, TFDistilBertModel, DistilBertConfig, TFBertModel, BertConfig
import tensorflow as tf
import numpy as np
import math
import argparse
import os

script_dir = os.path.dirname(__file__)
track_checkpoint_fname = os.path.join(script_dir, 'track_checkpoint.txt')
weights_dir = os.path.join(script_dir,"model_checkpts")

def parseArguments():
    parser = argparse.ArgumentParser()
    ### ONLY USE ONE OF (--load_weights, --save_weights, --test_gui) ###
    parser.add_argument("--load_weights", type=str, default="deadbeef") 
    # only if continuing to train on existing weights - if just testing use --test_only
    # also this will automatically save the weights back as well
    parser.add_argument("--save_weights", type=str, default="deafbeef")
    parser.add_argument("--bert", action="store_true")
    parser.add_argument("--test", type=str, default="deadbeef") 
    parser.add_argument("--from_titles", action="store_true") 
    # can't select both --save_weights and --test_only, it'll just train and save the weights
    # only select this if you want to reset training and not continue from the last checkpoint
    parser.add_argument("--test_gui", action="store_true")
    # if this is selected, will automatically load weights and start single-test interface
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_ft_epochs", type=int, default=5)
    parser.add_argument("--percent_data", type=float, default=1)
    parser.add_argument("--validation_split", type=float, default=0.1)
    # will probably need more than 10 epochs in final training
    parser.add_argument("--max_num_tokens", type=int, default=512)
    # maximum number of tokens considered in each input abstract (for both training & testing)
    args = parser.parse_args()
    return args

def batch_encode(tokenizer, texts, batch_size=256, max_length=512):
    """""""""
    A function that encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks that are ready to be fed 
    into a pre-trained transformer model.
    
    Input:
        - tokenizer:   Tokenizer object from the PreTrainedTokenizer Class
        - texts:       List of strings where each string represents a text
        - batch_size:  Integer controlling number of texts in a batch
        - max_length:  Integer controlling max number of words to tokenize in a given text
    Output:
        - input_ids:       sequence of texts encoded as a tf.Tensor object
        - attention_mask:  the texts' attention mask encoded as a tf.Tensor object
    """""""""
    input_ids = []
    attention_mask = []
    
    num_batches = math.floor(len(texts) / batch_size)
    for i in range(num_batches):
        batch = texts[i*batch_size:(i+1)*batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length,
                                             padding='max_length', 
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_token_type_ids=False
                                             )
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])
    
    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)


def build_model(transformer, max_length=512):
    """""""""
    Template for building a model off of the BERT or DistilBERT architecture
    for a binary classification task.
    
    Input:
      - transformer:  a base Hugging Face transformer model object (BERT or DistilBERT)
                      with no added classification head attached.
      - max_length:   integer controlling the maximum number of encoded tokens 
                      in a given sequence.
    
    Output:
      - model:        a compiled tf.keras.Model with added classification layers 
                      on top of the base pre-trained model architecture.
    """""""""
    
    # Define weight initializer with a random seed to ensure reproducibility
    weight_initializer = tf.keras.initializers.GlorotNormal() 
    
    # Define input layers
    input_ids_layer = tf.keras.layers.Input(shape=(max_length,), 
                                            name='input_ids', 
                                            dtype='int32')
    input_attention_layer = tf.keras.layers.Input(shape=(max_length,), 
                                                  name='input_attention', 
                                                  dtype='int32')
    
    # DistilBERT outputs a tuple where the first element at index 0
    # represents the hidden-state at the output of the model's last layer.
    # It is a tf.Tensor of shape (batch_size, sequence_length, hidden_size=768).
    last_hidden_state = transformer([input_ids_layer, input_attention_layer])[0]
    
    # We only care about DistilBERT's output for the [CLS] token, 
    # which is located at index 0 of every encoded sequence.  
    # Splicing out the [CLS] tokens gives us 2D data.
    cls_token = last_hidden_state[:, 0, :]
    
    ##                                                 ##
    ## Define additional dropout and dense layers here ##
    ##                                                 ##
    output = tf.keras.layers.Dense(512, kernel_initializer=weight_initializer)(cls_token)
    output = tf.keras.layers.LeakyReLU()(output)
    output = tf.keras.layers.Dense(256, kernel_initializer=weight_initializer)(output)
    output = tf.keras.layers.LeakyReLU()(output)
    output = tf.keras.layers.Dense(50, kernel_initializer=weight_initializer)(output)
    output = tf.keras.layers.LeakyReLU()(output)
    
    # Define a single node that makes up the output layer (for binary classification)
    output = tf.keras.layers.Dense(2, 
                                   activation='sigmoid',
                                   kernel_initializer=weight_initializer,  
                                   )(output)
    
    # Define the model
    model = tf.keras.Model([input_ids_layer, input_attention_layer], output)
    
    # Compile the model
    model.compile(tf.keras.optimizers.Adam(0.0001), 
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    
    return model

## NOTE: should we defined get_config and from_config functions to specify saving/loading weights?
# for that matter, do we need to save the compilatoin information & optimizer state? it's helpful
# if we are continuing to train the model, but not otherwise

def train(model, bert_model, train_ids, train_attn, train_labels, args):
    train_history1 = model.fit(
        x = [train_ids, train_attn],
        y = train_labels.numpy(),
        epochs = args.num_epochs,
        batch_size = args.batch_size,
        validation_split=args.validation_split,
        verbose=2
    )

    # finetune
    # if(bert_model != None):
    #     # Unfreeze distilBERT layers and make available for training
    #     for layer in bert_model.layers:
    #         layer.trainable = True

    #     # Recompile model after unfreezing
    #     model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), 
    #             loss=tf.keras.losses.BinaryCrossentropy(),
    #             metrics=['accuracy'])

    #     # Train the model
    #     train_history2 = model.fit(
    #         x = [train_ids, train_attn],
    #         y = train_labels.numpy(),
    #         epochs = args.num_ft_epochs,
    #         batch_size = args.batch_size,
    #         validation_split=args.validation_split,
    #         verbose=2
    #     )

def test(model, test_abstracts, test_labels, args):
    print("testing")
    if args.bert: tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    else: tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    test_abstracts_decoded = [s.decode('utf-8') for s in test_abstracts.numpy().tolist()]
    test_ids, test_attn = batch_encode(tokenizer, test_abstracts_decoded, args.batch_size, args.max_num_tokens)
    outputs = model([test_ids, test_attn])

    metric = tf.keras.metrics.CategoricalAccuracy()
    metric.update_state(test_labels, outputs)
    return metric.result().numpy() # return accuracy on training data

def test_one(model, test_abstract, args):
    if args.bert: tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    else: tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    test_id, test_attn = batch_encode(tokenizer, [test_abstract], 1, args.max_num_tokens)
    output = model([test_id, test_attn])
    return output # index that should have the higher value: 0 if human, 1 if chatgpt

def main(args):
    print("main function started")
    # load data
    script_dir = os.path.dirname(__file__)
    if args.from_titles:
        train_path = os.path.join(script_dir, "data/from_titles/train.csv")
        test_path = os.path.join(script_dir, "data/from_titles/test.csv")
    else:
        train_path = os.path.join(script_dir, "data/rephrased/train.csv")
        test_path = os.path.join(script_dir, "data/rephrased/test.csv")
    train_abstracts, train_labels, test_abstracts, test_labels = get_data(train_path, test_path)

    num_train = math.floor(train_labels.shape[0]*args.percent_data)
    train_abstracts = train_abstracts[:num_train]
    train_labels = train_labels[:num_train]
    num_test = math.floor(test_labels.shape[0]*args.percent_data)
    test_abstracts = test_abstracts[:num_test]
    test_labels = test_labels[:num_test]

    print("Data loaded")
    print("train_labels.shape:", train_labels.shape)
    print("test_labels.shape:", test_labels.shape)
    print("train abs.shape:", train_abstracts.shape)
    print("test avs.shape:", test_abstracts.shape)
   

    # read what number the last training run was (this is unnecessary if we're not testing or saving/loading weights)
    with open(track_checkpoint_fname, 'r') as f:
        last_checkpoint = int(f.read().strip())
    last_checkpoint_fname = f"model{last_checkpoint}"

    if args.test != "deadbeef":
        model = tf.keras.models.load_model(os.path.join(weights_dir, args.test))
        accuracy = test(model, test_abstracts, test_labels, args)
        print("Testing accuracy: ", accuracy)
    elif not args.test_gui:
        if args.bert: tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else: tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        train_abstracts_decoded = [s.decode('utf-8') for s in train_abstracts.numpy().tolist()]
        train_ids, train_attn = batch_encode(tokenizer, train_abstracts_decoded, args.batch_size, args.max_num_tokens)
        # load or create new model
        if args.load_weights != "deadbeef":
            if args.load_weights == "":
                model_load_name = last_checkpoint_fname
            else:
                model_load_name = args.load_weights
            print("path: ", os.path.join(weights_dir, model_load_name))
            model = tf.keras.models.load_model(os.path.join(weights_dir, model_load_name))
            bert_model = None
        else:
            if args.bert: 
                config = BertConfig(output_hidden_states=True)
                bert_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)
            else: 
                config = DistilBertConfig(output_hidden_states=True)
                bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config)

            for layer in bert_model.layers:
                layer.trainable = False

            model = build_model(bert_model, args.max_num_tokens)
        # *** train model ***
        train(model, bert_model, train_ids, train_attn, train_labels, args)
        # save model if necessary
        if args.save_weights != "deadbeef" or args.load_weights != "deadbeef":
            # update the number of the latest training checkpoint
            with open(track_checkpoint_fname, 'w') as f:
                f.write(str(last_checkpoint + 1))
            if args.save_weights == "" or args.load_weights == "":
                model_save_name = f"model{last_checkpoint + 1}"
            elif args.save_weights != "deadbeef":
                model_save_name = args.save_weights
            elif args.load_weights != "deadbeef":
                model_save_name = args.load_weights + "-1"
            else:
                model_save_name = f"model{last_checkpoint + 1}"
            model.save(os.path.join(weights_dir, model_save_name))
            print("Saved new weights in", model_save_name)
        # test model
        accuracy = test(model, test_abstracts, test_labels, args)
        print("Testing accuracy: ", accuracy)
    else:
        # run simple gui interface so you can try it yourself!
        # TODO: make the interface nice with tkinter
        test_abstract = input("Paste an abstract here: ").strip()
        model = tf.keras.models.load_model(os.path.join(weights_dir, last_checkpoint_fname))
        guess = test_one(model, test_abstract, args)
        print(f"This is probably written by {np.array(['a human', 'chatgpt'])[tf.math.argmax(guess)]} with probabilities [human, chatgpt] = {tf.nn.softmax(guess[0])}")

if __name__ == "__main__":
    args = parseArguments()
    main(args)