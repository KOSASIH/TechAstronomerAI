def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0  # Normalize pixel values
    return image

image_path = "path_to_image.jpg"  # Replace with the actual image path
image = load_image(image_path)
