import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

data_dir = 'C:/Users/ssumi/Desktop/training/'

IMG_SIZE = 100

categories = ['eye', 'ear', 'nose', 'whiskers', 'cheek']
num_classes = 3  # 0, 1, 2 점

data = load_data()

X = np.array([i[0] for i in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array([i[1] for i in data])

print(np.unique(y))  # [0 1 2 3 4]

y = np.where(y > 2, 2, y)  # 3, 4를 2로 변환

print(np.unique(y))  # [0 1 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=2)

def predict_new_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Unable to read image file")
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(img)
    return prediction[0]  

def show_predicted_results(image_paths):
    scores = []
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        for category in categories:
            if category in filename:
                class_name = category
                break
        else:
            print(f"Error: Unable to determine category for image {filename}")
            continue
        
        prediction = predict_new_image(image_path)
        if prediction is not None:
            class_index = categories.index(class_name)
            class_score = np.argmax(prediction)
            print(f"Predicted score for {class_name}: {class_score}")
            scores.append(class_score)
    if scores:
        average_score = sum(scores) / len(scores)
        print(f"Average score: {average_score}")

new_image_paths = ['C:/Users/ssumi/Desktop/test2/VNS/eye.png',
                   'C:/Users/ssumi/Desktop/test2/VNS/ear.png',
                   'C:/Users/ssumi/Desktop/test2/VNS/nose.png',
                   'C:/Users/ssumi/Desktop/test2/VNS/cheek.png',
                   'C:/Users/ssumi/Desktop/test2/VNS/whiskers.png']

show_predicted_results(new_image_paths)
