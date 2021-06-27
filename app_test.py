import requests
import json
import pickle
import sys

text = "please cal"

with open('input_lang.pkl', 'rb') as input1:
    input_lang = pickle.load(input1)
with open('output_lang.pkl', 'rb') as target:
    target_lang = pickle.load(target)
with open('output_lang1.pkl', 'rb') as target1:
    lang_target = pickle.load(target1)


import numpy as np
len_target = 23
len_input = 23

def sentence_to_vector(sentence, lang):
    pre = sentence
    vec = np.zeros(len_input)
    sentence_list = [lang[s] for s in pre.split(' ')]
    for i,w in enumerate(sentence_list):
        vec[i] = w
    return vec

# Given an input string, an encoder model (infenc_model) and a decoder model (infmodel),
def translate(input_sentence):
    sv = sentence_to_vector(input_sentence, input_lang)
    #sv = sv.reshape(1,len(sv))
    output_sentence = ""
    print(sv.shape)
    print(sv)
                  
    payload = {
            "instances":[{"input_image": sv.tolist()}]
    }
    try:
      r = requests.post('http://localhost:9000/v1/models/encoder_model:predict', json=payload)
      r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(r)
        print(r.content)
        return "Error: " + str(e)
        #json.loads(r.content)
        #sys.exit(1)
    epred= json.loads(r.content)['predictions']
    emb_out = epred[0]['bidirectional_3/concat:0']
    sh = epred[0]['concatenate_6/concat:0']
    sc = epred[0]['concatenate_7/concat:0']
    #[emb_out, sh, sc] = loaded_enc_model.predict(x=sv)
    print(epred[0].keys())
    i = 0
    start_vec = target_lang["<start>"]
    stop_vec = target_lang["<end>"]

    cur_vec = np.zeros((1))
    cur_vec[0] = start_vec
    cur_word = "<start>"
    output_sentence = ""
    print(cur_vec.shape)

    while cur_word != "<end>" and i < (len_target-1):
        i += 1
        if cur_word != "<start>":
            output_sentence = output_sentence + " " + cur_word
        #x_in = [cur_vec,sh, sc]

        ####
        payload = {
            "instances":[{
                          "inf_decoder_inputs:0": cur_vec.tolist(),
                          "state_input_c:0"     : sh,
                          "state_input_h:0"     : sc
                        }
                        ]
         }


        try:
          r = requests.post('http://localhost:9001/v1/models/decoder_model:predict', json=payload)
          r.raise_for_status()
        except requests.exceptions.HTTPError as e:
          print(r)
          print(r.content)
          return "Error: " + str(e)
        ####
        dpred= json.loads(r.content)['predictions']
        print(dpred[0].keys())
        nvec = dpred[0]['dense_7/truediv:0']
        sh   = dpred[0]['lstm_1/while/Exit_2:0']
        sc   = dpred[0]['lstm_1/while/Exit_3:0']
        #[nvec, sh, sc] = loaded_dec_model.predict(x=x_in)

        cur_vec[0] = np.argmax(nvec[0])
        cur_word = lang_target[cur_vec[0]]

    return output_sentence

prediction = translate(text.lower())
print(prediction)