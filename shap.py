from transformers import AutoTokenizer, TFDistilBertModel, DistilBertConfig, TFBertModel, BertConfig
import tensorflow as tf
import numpy as np
import shap
import os

# note: no argparse, args are hardcoded such that max length is always 512 tokens

script_dir = os.path.dirname(__file__)
track_checkpoint_fname = os.path.join(script_dir, 'track_checkpoint.txt')
weights_dir = os.path.join(script_dir,"model_checkpts")
test_path = os.path.join(script_dir, "data/from_titles/test.csv")
# find latest model
with open(track_checkpoint_fname, 'r') as f:
	last_checkpoint = int(f.read().strip())
last_checkpoint_fname = f"model{last_checkpoint}"

def get_test_data(test_path):
	test_abstracts = []
	test_labels = []
	total_skipped_pp_test = 0
	total_added_pp_test = 0

	with open(test_path) as test_csv:
		for line in csv.reader(test_csv):
			try:
				test_abstracts.append(line[0])
				test_labels.append(int(line[1]))
				total_added_pp_test += 1
			except:
				total_skipped_pp_test += 1
	
	print("total_skipped_pp_test", total_skipped_pp_test)
	print("total_added_pp_test", total_added_pp_test)

	test_abstracts = tf.convert_to_tensor(test_abstracts)
	test_labels = tf.convert_to_tensor(test_labels)
	test_labels = tf.one_hot(test_labels, 2)
	return test_abstracts, test_labels

test_abstracts, test_labels = get_test_data(test_path)
test_data_formatted = {'text' : test_abstracts, 'labels' : test_labels} # I think this should work?

# move to global
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
distil_bert = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
model = tf.keras.models.load_model(os.path.join(weights_dir, last_checkpoint_fname))

def predict(a_model, a_tokenizer, a_llm, test_abstract):
    # don't need to decode abstract as utf-8 because if it's input from the user it shouldn't be a bytestring
    tokenized_abstract = a_tokenizer([test_abstract], return_tensors='tf', max_length=512, \
                                   padding='max_length', truncation=True) # default max_length 512
    hidden_states = a_llm(tokenized_abstract).last_hidden_state
    hidden_states = tf.reshape(hidden_states, (hidden_states.shape[0], -1))
    output = a_model(hidden_states)
    return output # index that should have the higher value: 0 if human, 1 if chatgpt

explainer = shap.Explainer(lambda x : predict(model, tokenizer, distil_bert, x), tokenizer)

shap_values = explainer(test_data_formatted[:10], fixed_context=1, batch_size=2)