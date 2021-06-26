import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

'''
Initialize model from JSON and reload weights
1. load json and create model
'''
# TODO: settings.py/config file
MODEL_DIR = os.sep + "sample_data" + os.sep

with open(MODEL_DIR + "dec_model_num.json", "rb") as dec_json_file:
    dec_model_json = dec_json_file.read()
    dec_model = model_from_json(dec_model_json)

dec_json_file.close()

with open(MODEL_DIR + "enc_model_num.json", "rb") as enc_json_file:
    enc_model_json = enc_json_file.read()
    enc_model = model_from_json(enc_model_json)

enc_json_file.close()

### 2.load weights into new model
dec_model.load_weights(MODEL_DIR + "dec_model_num.h5")
enc_model.load_weights(MODEL_DIR + "enc_model_num.h5")
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
        inputs={'input_image': enc_model.input},
        outputs={t.name:t for t in enc_model.outputs})
    tf.saved_model.simple_save(
        sess,
        './decoder_model/1',
        inputs ={t.name:t for t in dec_model.input},
        outputs={t.name:t for t in dec_model.outputs})


print("loaded")

#%% Model test inference functions
# Converts the given sentence (just a string) into a vector of word IDs
# Output is 1-D: [timesteps/words]

def catch(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        pass

def sentence_to_vector(sentence, lang):

    pre = sentence
    vec = np.zeros(len_input)
    sentence_list = [catch(lambda: lang.word2idx[s]) for s in pre.split(' ')]
    for i,w in enumerate(sentence_list):
        vec[i] = w
    return vec

# Given an input string, an encoder model (infenc_model) and a decoder model (infmodel),
def translate(input_sentence, infenc_model, infmodel):
    sv = sentence_to_vector(input_sentence, input_lang)
    sv = sv.reshape(1,len(sv))
    [emb_out, sh, sc] = infenc_model.predict(x=sv)
    
    i = 0
    start_vec = target_lang.word2idx["<start>"]
    stop_vec = target_lang.word2idx["<end>"]
    
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
        cur_word = target_lang.idx2word[np.argmax(nvec[0,0])]
    return output_sentence

print ("Model testing functions loaded...")
#%% Test Inference
#Note that only words that we've trained the model on will be available, otherwise you'll get an error.

print ("Running test inference...")
test = [
    'hi there',
    'hell',
    'presentation please fin',
    'resignation please find at',
    'resignation please ',
    'have a nice we',
    'let me ',
    'promotion congrats ',
    'christmas Merry ',
    'please rev',
    'please ca',
    'thanks fo',
    'Let me kno',
    'Let me know if y',
    'this soun',
    'is this call going t'
]
  

import pandas as pd
output = []  
for t in test:  
  output.append({"Input seq":t.lower(), "Pred. Seq":translate(t.lower(), enc_model, dec_model)})

results_df = pd.DataFrame.from_dict(output) 
results_df.head(len(test))

