import numpy as np
import urllib.request
import cv2
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image

def main():

    class_names = ['Early blight', 'Late blight', 'Healthy']

    st.title('Predicting an Image based on 3 different categories')

    st.markdown("This is a model based on Tensorflow and Keras for classification of potato plant images.")
    st.markdown("Depending on the image you upload, the model will try to categorize it in one of 3 options available.")

    img_early_blight = Image.open('assets/early_blight.jpg')
    img_late_blight = Image.open('assets/late_blight.jpg')
    img_healthy = Image.open('assets/healthy.jpg')

    col1, col2, col3 = st.columns(3)

    col1.header("Early Blight")
    col1.image(img_early_blight)

    col2.header("Late Blight")
    col2.image(img_late_blight)

    col3.header("Healthy")
    col3.image(img_healthy)

    st.markdown("Upload an image similar to the ones mentioned above, the image will be resized and showed to you along with two possible classifications")

    upload_image = st.file_uploader("Choose a File",type=["png","jpg","jpeg"])

    model_xception = load_pretrained_model()

    batch_size = 32
    if upload_image is not None:
        bytes_data = upload_image.getvalue()
        data = np.frombuffer(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(data,1)
        img = cv2.resize(img, (150, 150))  # Resize the image
        st.image(img)
        np_img = np.array(img)
        np_img = np_img[np.newaxis, ...]
        img_dataset = tf.data.Dataset.from_tensor_slices(np_img)
        img_dataset = img_dataset.map(preprocess_prediction).batch(batch_size)
        prediction = model_xception.predict(img_dataset)
        prediction = np.ravel(prediction)
        idx_best_pred = prediction.argsort()[-1]
        st.markdown("The classifications is:")
        st.markdown(class_names[idx_best_pred])


def preprocess_prediction(image):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image

@st.cache(allow_output_mutation=True)
def load_pretrained_model():
    urllib.request.urlretrieve(
        'https://1214-project-potato-disease.s3.us-east-2.amazonaws.com/models/model_potato.h5', 'model_potato.h5')
    model = keras.models.load_model('./model_potato.h5')
    return model

if __name__ == '__main__':
    main()