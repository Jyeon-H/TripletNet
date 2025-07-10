import tensorflow as tf

def triplet_loss(y_true, y_pred):
    embed_size = tf.shape(y_pred)[1] // 3
    a = y_pred[:, :embed_size]
    p = y_pred[:, embed_size:2*embed_size]
    n = y_pred[:, 2*embed_size:]
    
    p_dist = tf.reduce_sum(tf.square(a - p), axis=1)
    n_dist = tf.reduce_sum(tf.square(a - n), axis=1)
    
    loss = tf.maximum(p_dist - n_dist + 1.0, 0.0)
    return tf.reduce_mean(loss)