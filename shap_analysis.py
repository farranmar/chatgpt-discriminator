from transformers import AutoTokenizer, TFDistilBertModel, DistilBertConfig, TFBertModel, BertConfig
import tensorflow as tf
import numpy as np
import shap
import os
import csv
import argparse

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_weights", type=str, default="deadbeef") 
    parser.add_argument("--bert", action="store_true")
    parser.add_argument("--from_titles", action="store_true") 
    parser.add_argument("--num_abstracts", type=float, default=10)
    parser.add_argument("--max_num_tokens", type=int, default=512)
    args = parser.parse_args()
    return args

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


def predict(a_model, a_tokenizer, a_llm, test_abstract, args):
    # don't need to decode abstract as utf-8 because if it's input from the user it shouldn't be a bytestring
    tokenized_abstract = a_tokenizer([test_abstract], return_tensors='tf', max_length=args.max_num_tokens, \
                                   padding='max_length', truncation=True) # default max_length 512
    hidden_states = a_llm(tokenized_abstract).last_hidden_state
    hidden_states = tf.reshape(hidden_states, (hidden_states.shape[0], -1))
    output = a_model(hidden_states)
    return output # index that should have the higher value: 0 if human, 1 if chatgpt

def main(args):
	script_dir = os.path.dirname(__file__)
	track_checkpoint_fname = os.path.join(script_dir, 'track_checkpoint.txt')
	weights_dir = os.path.join(script_dir,"model_checkpts")
	if args.from_titles: test_path = os.path.join(script_dir, "data/from_titles/test.csv")
	else: test_path = os.path.join(script_dir, "data/rephrased/test.csv")
	# find latest model
	with open(track_checkpoint_fname, 'r') as f:
		last_checkpoint = int(f.read().strip())
	last_checkpoint_fname = f"model{last_checkpoint}"
	if args.load_weights == "" or args.load_weights == "deadbeef":
		model_load_name = last_checkpoint_fname
	else:
		model_load_name = args.load_weights
	print("model=tf.keras.models.load_model(", os.path.join(weights_dir, model_load_name), ")")
	model = tf.keras.models.load_model(os.path.join(weights_dir, model_load_name))

	test_abstracts, test_labels = get_test_data(test_path)
	test_abstracts_decoded = [s.decode('utf-8') for s in test_abstracts.numpy().tolist()]

	if args.bert: tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
	else: tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
	if args.bert: 
		config = BertConfig(output_hidden_states=True)
		bert_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)
	else: 
		config = DistilBertConfig(output_hidden_states=True)
		bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config)

	explainer = shap.DeepExplainer(lambda x : predict(model, tokenizer, bert_model, x, args), tokenizer)

	shap_values = explainer(test_abstracts_decoded[:10], fixed_context=1, batch_size=2)

	shap.plots.waterfall(shap_values[0])


if __name__ == "__main__":
    args = parseArguments()
    main(args)