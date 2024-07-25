import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model_path = 'C:/Users/idofr/Documents/ssd/'
model = tf.saved_model.load(model_path)


def load_image_into_numpy_array(path):
    return np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))


def detect_objects(image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    model_fn = model.signatures['serving_default']
    detections = model_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    return detections


def main():
    image_path = 'Dog2.jpeg'
    image = load_image_into_numpy_array(image_path)

    detections = detect_objects(image)

    classes = detections['detection_classes'].astype(np.int64)
    scores = detections['detection_scores']

    unique, counts = np.unique(classes, return_counts=True)

    max_count_index = np.argmax(counts)
    most_common_class = unique[max_count_index]
    most_common_count = counts[max_count_index]
    most_common_score = scores[classes == most_common_class]

    print(
        f"Most common object: {most_common_class} appears {most_common_count} times with average score {np.mean(most_common_score)}")

    for i in range(len(classes)):
        print(f"Object {classes[i]} detected with score {scores[i]}")


if __name__ == "__main__":
    main()
