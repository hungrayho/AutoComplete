import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

'''
Initialize model from JSON and reload weights
1. load json and create model
'''
with open("dec_model_num.json", "rb") as dec_json_file:
    loaded_dec_model_json = dec_json_file.read()
    loaded_dec_model = model_from_json(loaded_dec_model_json)

with open("enc_model_num.json", "rb") as enc_json_file:
    loaded_enc_model_json = enc_json_file.read()
    loaded_enc_model = model_from_json(loaded_enc_model_json)

enc_json_file.close()
dec_json_file.close()

### 2.load weights into new model
loaded_dec_model.load_weights("dec_model_num.h5")
loaded_enc_model.load_weights("enc_model_num.h5")
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
        inputs={'input_image': loaded_enc_model.input},
        outputs={t.name:t for t in loaded_enc_model.outputs})
    tf.saved_model.simple_save(
        sess,
        './decoder_model/1',
        inputs ={t.name:t for t in loaded_dec_model.input},
        outputs={t.name:t for t in loaded_dec_model.outputs})


print("loaded")
