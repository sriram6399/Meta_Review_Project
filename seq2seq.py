import datasets  
import pandas as pd  
import numpy  
import matplotlib.pyplot as plt  
import re
import nltk  
import contractions  
import utilities
import gensim  
from tensorflow.keras import callbacks, models, layers,optimizers, preprocessing as kprocessing 

def lstm_model(X_train,X_dic_vocabulary,y_dic_vocabulary):
    lstm_units = 200
    embeddings_size = 300
    x_in = layers.Input(name="x_in", shape=(X_train.shape[1],))
    layer_x_emb = layers.Embedding(name="x_emb", input_dim=len(X_dic_vocabulary),output_dim=embeddings_size, trainable=True)
    x_emb = layer_x_emb(x_in)

    layer_x_lstm = layers.LSTM(name="x_lstm", units=lstm_units, dropout=0.4, return_sequences=True, return_state=True)
    x_out, state_h, state_c = layer_x_lstm(x_emb)

    y_in = layers.Input(name="y_in", shape=(None,))

    layer_y_emb = layers.Embedding(name="y_emb", input_dim=len(y_dic_vocabulary), output_dim=embeddings_size, trainable=True)
    y_emb = layer_y_emb(y_in)
 
    layer_y_lstm = layers.LSTM(name="y_lstm", units=lstm_units, dropout=0.4, return_sequences=True, return_state=True)
    y_out, _, _ = layer_y_lstm(y_emb, initial_state=[state_h, state_c])

    layer_dense = layers.TimeDistributed(name="dense",layer=layers.Dense(units=len(y_dic_vocabulary), activation='softmax'))
    y_out = layer_dense(y_out)


    model = models.Model(inputs=[x_in, y_in], outputs=y_out,name="Seq2Seq")
    adam = optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def model_train(X_train,y_train,model):
    training = model.fit(x=[X_train, y_train[:,:-1]], y=y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:,1:],
                     batch_size=1, 
                     epochs=20, 
                     shuffle=True, 
                     verbose=1, 
                     validation_split=0.3,
                     callbacks=[callbacks.EarlyStopping(
                                monitor='val_loss', 
                                mode='min', verbose=1, patience=3)]
                      )

    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_'+metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    plt.savefig('Figures/training_accuracy.jpg')
    model.save('models/trained_model.h5')
    return training,model

stopwords = nltk.corpus.stopwords.words("english")

data_train= utilities.read_data('Data/train_data.csv')
data_test= utilities.read_data('Data/test_data.csv')

data_train["X"] = data_train["review"].apply(lambda x: utilities.utils_preprocess_text(x, punkt=True, lower=True, slang=True, stopwords=stopwords, stemm=False, lemm=True))
data_train["Y"] = data_train["meta_review"].apply(lambda x: utilities.utils_preprocess_text(x, punkt=True, lower=True, slang=True, stopwords=stopwords, stemm=False, lemm=True))
data_test["X"] = data_test["review"].apply(lambda x: utilities.utils_preprocess_text(x, punkt=True, lower=True, slang=True, stopwords=stopwords, stemm=False, lemm=True))
data_test["Y"] = data_test["meta_review"].apply(lambda x: utilities.utils_preprocess_text(x, punkt=True, lower=True, slang=True, stopwords=stopwords, stemm=False, lemm=True))
special_tokens = ("<START>", "<END>")
data_train["Y"] = data_train['Y'].apply(lambda x: special_tokens[0]+' '+x+' '+special_tokens[1])
data_test["Y"] = data_test['Y'].apply(lambda x: special_tokens[0]+' '+x+' '+special_tokens[1])

x_tokenizer = utilities.get_tokenizer(data_train["X"])
x_dic_vocabulary = {"<PAD>":0}
x_dic_vocabulary.update(x_tokenizer.word_index)
y_tokenizer = utilities.get_tokenizer(data_train["Y"])
y_dic_vocabulary = {"<PAD>":0}
y_dic_vocabulary.update(y_tokenizer.word_index)

X_text2seq= x_tokenizer.texts_to_sequences(data_train["X"])
X_train = kprocessing.sequence.pad_sequences(X_text2seq, maxlen=50, padding="post", truncating="post")
X_test_text2seq= x_tokenizer.texts_to_sequences(data_test["X"])
X_test = kprocessing.sequence.pad_sequences(X_test_text2seq, maxlen=50, padding="post", truncating="post")
Y_text2seq= y_tokenizer.texts_to_sequences(data_train["Y"])
Y_train = kprocessing.sequence.pad_sequences(Y_text2seq, maxlen=50, padding="post", truncating="post")
Y_test_text2seq= y_tokenizer.texts_to_sequences(data_test["Y"])
Y_test = kprocessing.sequence.pad_sequences(Y_test_text2seq, maxlen=50, padding="post", truncating="post")

X_embeddings = utilities.get_embedding_vector(x_dic_vocabulary)
Y_embeddings = utilities.get_embedding_vector(y_dic_vocabulary)
model = lstm_model(X_train, x_dic_vocabulary,y_dic_vocabulary)
training,model = model_train(X_train,Y_train,model)