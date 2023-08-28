import os
import pandas as pd
import numpy as np
import librosa
import pickle
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def feature_extractor(file):
    audio, sample_rate = librosa.load(file, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

def build_and_train_model(metadata_file, data_path):
    metadata = pd.read_csv(metadata_file)

    extracted_features = []
    for index_num, row in metadata.iterrows():
        file_name = os.path.join(data_path, str(row['filename']))
        final_class_labels = row['id']
        data = feature_extractor(file_name)
        extracted_features.append([data, str(final_class_labels)])

    extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
    X = np.array(extracted_features_df['feature'].tolist())
    y = np.array(extracted_features_df['class'].tolist())

    # Get the number of unique labels
    n_labels = len(np.unique(y))

    # Convert integer labels to one-hot encoded format
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = to_categorical(y, num_classes=n_labels)

    # Data Preprocessing
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Add one more dimension
    input_shape = (X.shape[1], X.shape[2])  # Update the input shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_labels, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    num_epochs = 100
    num_batch_size = 32

    checkpointer = ModelCheckpoint(filepath='a_c.hdf5', verbose=1, save_best_only=True)
    model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer])

    # Save the label encoder to disk
    label_encoder_filename = 'label_encoder.pkl'
    with open(label_encoder_filename, 'wb') as le_file:
        pickle.dump(label_encoder, le_file)

    return model, label_encoder, X_test, y_test

# Assuming you have the metadata file and data path
metadata_file = "D:/dattasir/bird_songs_metadata.csv"
data_path = "D:/dattasir/wavfiles"
trained_model, label_encoder, X_test, y_test = build_and_train_model(metadata_file, data_path)

# Save the trained model's architecture and weights to disk
trained_model.save("trained_model.h5")

# Making Predictions
tn = "D:/dattasir/wavfiles/217833-11.wav"
prediction_feature = feature_extractor(tn)
prediction_feature = prediction_feature.reshape(1, prediction_feature.shape[0], 1)

# Predict using the loaded model
prediction = trained_model.predict(prediction_feature)

# Get the predicted class index (maximum probability)
predicted_class_index = np.argmax(prediction)

# Convert the predicted class index back to class label using the loaded label encoder
predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]

# print("Predicted class name:", predicted_class_label)
label_encoder_filename = 'label_encoder.pkl'
with open(label_encoder_filename, 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Calculate metrics for the test dataset
y_pred = trained_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Convert the numeric class labels back to original class names
y_true_classes_labels = label_encoder.inverse_transform(y_true_classes)
y_pred_classes_labels = label_encoder.inverse_transform(y_pred_classes)

# Calculate metrics for the test dataset
accuracy = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
recall = recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
f1_score_value = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)

print("Class Labels:", label_encoder.classes_)
print("True Class Labels:", y_true_classes_labels)
print("Predicted Class Labels:", y_pred_classes_labels)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score_value)
print("Predicted class name:", predicted_class_label)
