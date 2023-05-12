# NotChatGPT

welcome to our DL final project :)

<br> 

## Running the model
To run the model, run `python model.py`  
See the following optional arguments:  
`--load_weights`: takes a string representing the name of the model (in `model_checkpts`) that you want to load  
`--save_weights`: takes a string representing the name you want to give the model, which will be the name of the folder it is saved in in `model_checkpts`  
`--test`: will test the model instead of training; takes a string representing the name of the model (in `model_checkpts`) that you want to test  
`--test_gui`: will allow the user to provide an input, runs the model on that input and prints the output  
`--bert`: indicates the model should use BERT instead of distilBERT (note this will use a lot of time and memory)  
`--from_titles`: indicates you want to use the data generated from paper titles rather than rephrased from human-written abstracts  
`--batch_sze`: takes an int of the batch size you wish to use; defaults to 256
`--num_epochs`: takes an int of the number of epochs you want to train the model for; defaults to 10  
`--percent_data`: takes a float of the proportion of data you wish to use for training and testing; defaults to 1  
`--max_num_tokens`: takes an int of the max number of tokens to use when tokenizing inputs; defaults to 512




