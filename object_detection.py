def detect_objects(image):
    resized_image = cv2.resize(image, (224, 224))
    expanded_image = np.expand_dims(resized_image, axis=0)
    preprocessed_image = tf.keras.applications.resnet50.preprocess_input(expanded_image)

    predictions = model.predict(preprocessed_image)
    predicted_classes = tf.keras.applications.resnet50.decode_predictions(predictions, top=5)[0]

    return predicted_classes

detected_objects = detect_objects(image)
