import cv2
import streamlit as st
import joblib
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
 
# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X = mnist["data"]
y = mnist["target"].astype(np.uint8)
X_train = X[:60000]
y_train = y[:60000]
 
# Train a KNN classifier
knn_clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=3, weights='distance', metric='euclidean')
knn_clf.fit(X, y)
# Streamlit UI
st.title("Handwritten Digit Recognition App")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Funktion för att göra förutsägelser
if uploaded_files:
    for uploaded_file in uploaded_files:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR)
        flattened_image = resized_image.flatten().reshape(1, -1)

        lower_pixel = 100
        upper_pixel = 130
 
        for i in range(resized_image.shape[0]):
             for j in range(resized_image.shape[1]):
               if resized_image[i, j] <= lower_pixel:
                  resized_image[i, j] = 0
               elif resized_image[i, j] > upper_pixel:
                   resized_image[i, j] = 255
    
        # Gör förutsägelser med modellen
        prediction = knn_clf.predict(flattened_image)
        st.image(image, caption=f"Prediction: {prediction[0]}", use_column_width=True)




