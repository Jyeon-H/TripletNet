import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_data(image_dir, csv_path):
    image_path = sorted(glob.glob(image_dir + '/*.jpg'))
    sc = pd.read_csv(csv_path)
    score = sc[['filename', 'skill_mean']]
    images = preprocess_images(image_path)
    scores = preprocess_scores(score)
    return images, scores

def preprocess_images(image_paths):
    images = []
    for path in tqdm(image_paths):
        img = cv2.imread(path)
        img = cv2.resize(img, (224,224))
        images.append(img)
    return np.array(images)

def preprocess_scores(score_df):
    return np.array(score_df)

def triplet_generator(images, scores, batch_size=32):
    while True:
        a_idx = np.random.choice(len(images), batch_size, replace=False)
        a_scores = scores[a_idx]
        anchors = images[a_idx]

        p_idx, n_idx = [], []
        for a_score in a_scores:
            pos_range = (scores[:, 1] >= a_score[1] - 5.0) & (scores[:, 1] <= a_score[1] + 5.0)
            neg_range = (scores[:, 1] <= a_score[1] - 30.0) | (scores[:, 1] >= a_score[1] + 30.0)

            pos = np.where(pos_range)[0]
            neg = np.where(neg_range)[0]

            p_idx.append(np.random.choice(pos))
            n_idx.append(np.random.choice(neg))
        
        positives = images[p_idx]
        negatives = images[n_idx]

        yield [anchors, positives, negatives], np.zeros((batch_size, 1))