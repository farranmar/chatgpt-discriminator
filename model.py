from preprocess import get_data
from transformers import AutoTokenizer, TFDistilBertModel
import tensorflow as tf
import math

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
        print("putputs shape", outputs.shape)
        return outputs

    

def train(model, train_abstracts, train_labels):
    print("training")
    batch_size = 256
    num_batches = math.floor(train_abstracts.shape[0] / batch_size)
    indices = tf.random.shuffle(tf.range(train_abstracts.shape[0]))
    train_abstracts = tf.gather(train_abstracts, indices)
    train_labels = tf.gather(train_labels, indices)
    train_abstracts = train_abstracts[:batch_size*num_batches]
    train_labels = train_labels[:batch_size*num_batches]
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    distil_bert = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    # train_abstracts_list = train_abstracts.numpy().tolist()
    # train_abstracts_list = [s.decode('utf-8') for s in train_abstracts_list]
    # tokenized_abstracts = tokenizer(train_abstracts_list, return_tensors='tf', max_length=512, padding='max_length', truncation=True)
    # print("tokenized abstracts", type(tokenized_abstracts))
    total_loss = 0
    for i in range(num_batches):
        batch_abstracts = train_abstracts[i * batch_size:(i+1)*batch_size]
        batch_labels = train_labels[i * batch_size:(i+1)*batch_size]
        print("batcha bstract", batch_abstracts.dtype)
        batch_abstracts_list = batch_abstracts.numpy().tolist()
        batch_abstracts_list = [s.decode('utf-8') for s in batch_abstracts_list]
        tokenized_abstracts = tokenizer(batch_abstracts_list, return_tensors='tf', max_length=512, padding='max_length', truncation=True)
        hidden_states = distil_bert(tokenized_abstracts).last_hidden_state
        hidden_states = tf.reshape(hidden_states, (batch_size, -1))

        with tf.GradientTape() as tape:
            outputs = model(hidden_states)
            print("outputs", outputs.dtype)
            print("batch_lbables", batch_labels.dtype)
            loss = model.loss(outputs, batch_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss
    return total_loss / num_batches


def main():
    train_abstract, train_labels, test_abstract, test_labels = get_data('data/train.csv', 'data/test.csv')
    num_epochs = 10
    model = GPTClassifier()
    for i in range(num_epochs):
        total_loss = train(model, train_abstract, train_labels)
        print("Epoch", i, ": total_loss = ", total_loss)

if __name__ == "__main__":
    main()