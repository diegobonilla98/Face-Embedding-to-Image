import increase_memory_alloc

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

from model import FaceGenerator


face_embedding_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
face_detection_model = MTCNN()
gan = FaceGenerator(is_test=True)
generator = gan.generator
weights_path = './RESULTS/v3_best_gen.h5'
generator.load_weights(weights_path)

image_path = './test_images/rajoy.jpg'
image = cv2.imread(image_path)[:, :, ::-1]

face = face_detection_model.detect_faces(image)[0]
x, y, w, h = face['box']
face = image[y: y + h, x: x + w]
face = cv2.resize(face, (224, 224), cv2.INTER_LANCZOS4)
face_tensor = np.expand_dims(face.copy().astype('float32'), axis=0)
face_tensor = preprocess_input(face_tensor, version=2)
face_embedding = face_embedding_model.predict(face_tensor).reshape((1, 2048))

# face_embedding_draw = face_embedding.copy().reshape((32, 64)) / face_embedding.max()
# cv2.imwrite('./test_images/di_caprio_embedding.png', cv2.applyColorMap(np.uint8(face_embedding_draw * 255.), cv2.COLORMAP_JET))

result = generator.predict(face_embedding)
result = (result[0] + 1) / 2

plt.figure(0)
plt.imshow(face)
plt.figure(1)
plt.imshow(result)
plt.show()

p, e = os.path.splitext(image_path)
cv2.imwrite(p + '_reconstructed.jpg', np.uint8(result * 255.)[:, :, ::-1])
