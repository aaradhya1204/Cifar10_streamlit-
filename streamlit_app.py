import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from PIL import Image

def main():
    st.title('Cifar10 Web Classifier')
    st.write('Upload an image, and the model will predict its class.')

    file = st.file_uploader('Please upload an image', type=['jpg', 'png'])
    if file:
        image = Image.open(file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        resized_image = image.resize((32, 32))
        img_array = np.array(resized_image) / 255.0
        img_array = img_array.reshape((1, 32, 32, 3))

        model = tf.keras.models.load_model('cifar10_model.h5')

        predictions = model.predict(img_array)[0]
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        predicted_class = cifar10_classes[np.argmax(predictions)]

        st.write(f"Predicted Class: **{predicted_class}**")
        st.write("Prediction Probabilities:")
        for i, prob in enumerate(predictions):
            st.write(f"{cifar10_classes[i]}: {prob:.2f}")

        # Visualization
        fig, ax = plt.subplots()
        y_pos = np.arange(len(cifar10_classes))
        ax.barh(y_pos, predictions, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cifar10_classes)
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        ax.set_title('CIFAR10 Predictions')
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.text("You have not uploaded an image yet.")

if __name__ == '__main__':
    main()
