import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

def create_base_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    return Model(inputs=base_model.input, outputs=x)
    
def create_triplet_model():
    base_model = create_base_model()
    
    input_a = layers.Input(shape=(224, 224, 3))
    input_p = layers.Input(shape=(224, 224, 3))
    input_n = layers.Input(shape=(224, 224, 3))
    
    embed_a = base_model(input_a)
    embed_p = base_model(input_p)
    embed_n = base_model(input_n)
    
    output = layers.Concatenate()([embed_a, embed_p, embed_n])
    
    return Model(inputs=[input_a, input_p, input_n], outputs=output)