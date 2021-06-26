import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

from src.interface.metaclasses import meta_inference_engine


class inference_engine(meta_inference_engine):
    
    DEFAULT_DIR = "/model/"
    
    def __init__(self, MODEL_DIR = DEFAULT_DIR):
        # config
        self.MODEL_DIR = MODEL_DIR
        # TODO: settings/config

        # attributes
        self._enc_model = None
        self._dec_model = None
        self._target_lang = None
        

        @classmethod
        def load_lang_index(): # TODO: load word index
            with open(self.MODEL_DIR + 'language_index.pkl', 'rb') as pkl_file:

                self._target_lang = pickle.load(pkl_file)


        @classmethod
        def load_model():
            '''
            Initialize model from JSON and reload weights
            1. load json and create model
            '''
            # TODO: settings.py/config file

            with open(self.MODEL_DIR + "dec_model_num.json", "rb") as dec_json_file:
                dec_model_json = dec_json_file.read()
                self.dec_model = model_from_json(dec_model_json)

            dec_json_file.close()

            with open(self.MODEL_DIR + "enc_model_num.json", "rb") as enc_json_file:
                enc_model_json = enc_json_file.read()
                self.enc_model = model_from_json(enc_model_json)

            enc_json_file.close()

            ### 2.load weights into new model
            self.dec_model.load_weights(self.MODEL_DIR + "dec_model_num.h5")
            self.enc_model.load_weights(self.MODEL_DIR + "enc_model_num.h5")
            print("Loaded models from disk")

            tf.keras.backend.set_learning_phase(0) # Ignore dropout at inference

            '''
            Fetch the Keras session and save the model
            The signature definition is defined by the input and output tensors
            And stored with the default serving key
            '''

            with tf.keras.backend.get_session() as sess:
                tf.saved_model.simple_save(
                    sess,
                    './encoder_model/1',
                    inputs={'input_image': self.enc_model.input},
                    outputs={t.name:t for t in self.enc_model.outputs})
                tf.saved_model.simple_save(
                    sess,
                    './decoder_model/1',
                    inputs ={t.name:t for t in self.dec_model.input},
                    outputs={t.name:t for t in self.dec_model.outputs})


            print("loaded inference model")
        
        

    @staticmethod
    def __catch(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            pass

    @classmethod
    def sentence_to_vector(sentence, lang):
        """
        converts sentence to word vector
        """
        pre = sentence
        vec = np.zeros(len_input)
        sentence_list = [self.__catch(lambda: lang.word2idx[s]) for s in pre.split(' ')]
        for i,w in enumerate(sentence_list):
            vec[i] = w
        return vec

    @classmethod
    def translate(input_sequence, inf_enc_model, inf_model):
        """
        runs model inference on input sequence, returns output_sentence
        """
        sv = sentence_to_vector(input_sentence, input_lang)
        sv = sv.reshape(1,len(sv))
        [emb_out, sh, sc] = infenc_model.predict(x=sv)
        
        i = 0
        start_vec = self._target_lang.word2idx["<start>"]
        stop_vec = self._target_lang.word2idx["<end>"]
        
        cur_vec = np.zeros((1,1))
        cur_vec[0,0] = start_vec
        cur_word = "<start>"
        output_sentence = ""

        while cur_word != "<end>" and i < (len_target-1):
            i += 1
            if cur_word != "<start>":
                output_sentence = output_sentence + " " + cur_word
            x_in = [cur_vec, sh, sc]
            [nvec, sh, sc] = infmodel.predict(x=x_in)
            cur_vec[0,0] = np.argmax(nvec[0,0])
            cur_word = self._target_lang.idx2word[np.argmax(nvec[0,0])]
        return output_sentence

    @classmethod
    def predict(payload):
        try:
            return self.translate(payload.lower())
        except Error:
            return str(Error)
    
    self.load_lang_index()
    self.load_model()
