import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf


def process_image(uploaded_file):
    file_content = uploaded_file.read()
    nparr = np.frombuffer(file_content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


model = tf.saved_model.load("./satelite data model")


def perform_segmentation(image):
    resized_image = cv2.resize(image, (128, 128))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    normalized_image = resized_image / 255.0

    input_array = np.expand_dims(normalized_image, axis=0).astype(np.float32)
    segmented_mask = model(tf.constant(input_array))[0]

    threshold = 0.5
    binary_mask = (segmented_mask > threshold).numpy().astype(np.uint8)

    # Resize the binary mask to match the size of the input image
    binary_mask_resized = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))

    segmented_image = cv2.bitwise_and(image, image, mask=binary_mask_resized)

    return segmented_image


def main():
    st.title("Solution Challenge")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = process_image(uploaded_file)
        segmented_image = perform_segmentation(image)

        # Create two columns to display images side by side
        col1, col2 = st.columns(2)

        # Display original image in the first column
        col1.image(image, caption="Original Image", use_column_width=True)

        # Display segmented image in the second column
        col2.image(
            segmented_image, caption="Segmentation Results", use_column_width=True
        )


if __name__ == "__main__":
    main()
