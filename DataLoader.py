import glob
import cv2
import os
import matplotlib.pyplot as plt
import random
import keras_vggface
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np


class DataLoader:
    def __init__(self, image_shape, batch_size):
        self.image_size = image_shape[:2]
        self.batch_size = batch_size
        ROOT = '/media/bonilla/HDD_2TB_basura/databases/CelebA/archive/img_align_celeba/img_align_celeba'
        self.images = glob.glob(os.path.join(ROOT, '*.jpg'))
        random.shuffle(self.images)
        self.face_embedding_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    def _get_face_embedding(self, image_path, rand_args):
        image = cv2.imread(image_path)[40:190, 40:130, ::-1]
        image = cv2.resize(image, (224, 224), cv2.INTER_LANCZOS4)
        if rand_args[0] > 0.5:
            image = cv2.flip(image, 1)
        image = np.expand_dims(image.astype('float32'), axis=0)
        image = preprocess_input(image, version=2)
        return self.face_embedding_model.predict(image).flatten()

    def _process_image(self, image_path, rand_args):
        image = cv2.imread(image_path)[40:190, 40:130, ::-1]
        image = cv2.resize(image, self.image_size, cv2.INTER_LANCZOS4)
        if rand_args[0] > 0.5:
            image = cv2.flip(image, 1)
        return (image.astype('float32') - 127.5) / 127.5

    def load_batch(self, num=None):
        batch_images = random.sample(self.images, self.batch_size if num is not None else num)
        images = []
        embeddings = []
        for batch_image in batch_images:
            rand_args = np.random.rand(1)
            image = self._process_image(batch_image, rand_args)
            embedding = self._get_face_embedding(batch_image, rand_args)
            images.append(image)
            embeddings.append(embedding)
        return np.array(embeddings), np.array(images)
