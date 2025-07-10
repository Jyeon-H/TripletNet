import numpy as np
from data_loader import load_data, triplet_generator
from model import create_triplet_model
from loss import triplet_loss
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD
import os

def main():
    image_dir = "./data/images/*.jpg"
    score_csv = "./data/scores.csv"

    images, scores = load_data(image_dir, score_csv)

    train_img, test_img, train_score, test_score = train_test_split(images, scores, test_size=0.2, random_state=42)

    model = create_triplet_model()
    model.compile(optimizer=SGD(learning_rate=0.001), loss=triplet_loss)

    checkpoint = ModelCheckpoint(f"{model_dir}/tripletnet_checkpoint.h5", save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = model.fit(
        triplet_generator(train_img, train_score, batch_size=64),
        validation_data=triplet_generator(test_img, test_score, batch_size=64),
        steps_per_epoch=len(train_img) // 64,
        epochs=300,
        callbacks=[checkpoint, early_stopping]   
    )

    model.save(f"{model_dir}/triplet_model_final.h5")
    print("모델 저장 완료")

if __name__ == "__main__":
    main()