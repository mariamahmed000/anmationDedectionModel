import pandas as pd
from flask import Flask, jsonify, request
import pickle
import numpy as np
import pandas as pd
from keras.utils import pad_sequences

from model import tokenizer, label_encoder, max_sequence_length

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))


@app.route('/emotion', methods=['POST'])
def predict():
    data = request.json
    new_samples = data['samples']
    new_sequences = tokenizer.texts_to_sequences(new_samples)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)
    predicted_labels = model.predict(new_padded_sequences)
    predicted_emotions = label_encoder.inverse_transform(np.argmax(predicted_labels, axis=1))
    response = {'predictions': predicted_emotions.tolist()}
    return jsonify(response)



if __name__ == "__main__":
  app.run(host='0.0.0.0', port=8080)