# birdapp/views.py
import os
import librosa
import pickle
import numpy as np
from django.shortcuts import render
from django.conf import settings
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import pandas as pd

# Function to extract audio features using librosa
def feature_extractor(file):
    audio, sample_rate = librosa.load(file, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# View to handle audio file upload and prediction
def predict_audio(request):
    
    if request.method == 'POST' and request.FILES['audio_file']:
        audio_file = request.FILES['audio_file']

        # Save the uploaded audio to a temporary file
        temp_path = os.path.join(settings.BASE_DIR, 'birdapp', 'files', 'temp_audio.wav')
        with open(temp_path, 'wb') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        # Load the trained model and label encoder
        model_path = os.path.join(settings.BASE_DIR, 'birdapp', 'files', 'trained_model.h5')
        label_encoder_path = os.path.join(settings.BASE_DIR, 'birdapp', 'files', 'label_encoder.pkl')

        model = load_model(model_path)
        with open(label_encoder_path, 'rb') as le_file:
            label_encoder = pickle.load(le_file)

        # Extract features from the uploaded audio file
        prediction_feature = feature_extractor(temp_path)
        prediction_feature = prediction_feature.reshape(1, prediction_feature.shape[0], 1)

        # Predict using the loaded model
        prediction = model.predict(prediction_feature)

        # Get the predicted class index (maximum probability)
        predicted_class_index = np.argmax(prediction)

        # Convert the predicted class index back to class label using the loaded label encoder
        predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
        df_path = os.path.join(settings.BASE_DIR, 'birdapp', 'files', "bird_songs_metadata.csv")
        df = pd.read_csv(df_path)
        x=df.loc[df["id"]==int(predicted_class_label)]
        p=x.iloc[[0]].values.tolist()
        # print(p[0][2])

        # Render the result.html template with the predicted class name as a context variable
        return render(request, 'birdapp/result.html', {'predicted_class_name': p[0][2]})

    return render(request, 'birdapp/upload.html')
def home(request):
    return render(request, 'birdapp/index1.html')
def progress(request):
    return render(request, 'birdapp/progress.html')
def confusion_matrix_view(request):
    # Load the trained model and label encoder
    model_path = os.path.join(settings.BASE_DIR, 'birdapp', 'files', 'trained_model.h5')
    label_encoder_path = os.path.join(settings.BASE_DIR, 'birdapp', 'files', 'label_encoder.pkl')

    model = load_model(model_path)
    with open(label_encoder_path, 'rb') as le_file:
        label_encoder = pickle.load(le_file)

    # Load the test data and true labels (ground truth) if available
    test_data_path = os.path.join(settings.BASE_DIR, 'birdapp', 'files', 'test_data.pkl')
    true_labels_path = os.path.join(settings.BASE_DIR, 'birdapp', 'files', 'true_labels.pkl')

    with open(test_data_path, 'rb') as test_data_file:
        test_data = pickle.load(test_data_file)

    with open(true_labels_path, 'rb') as true_labels_file:
        true_labels = pickle.load(true_labels_file)

    # Make predictions on the test data
    predictions = []
    for audio_file in test_data:
        prediction_feature = feature_extractor(audio_file)
        prediction_feature = prediction_feature.reshape(1, prediction_feature.shape[0], 1)
        prediction = model.predict(prediction_feature)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
        predictions.append(predicted_class_label)

    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    return render(request, 'birdapp/confusion_matrix.html', {'confusion_matrix': cm})