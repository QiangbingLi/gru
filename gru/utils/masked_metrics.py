""" 
For training, you'll want to implement your own masked loss 
and accuracy functions
"""
import tensorflow as tf

def masked_loss(y_true, y_pred):
    """
    Calculate the masked loss for each item in the batch.
    y_true.shape: (batch, t)
    y_pred.shape: (batch, t, vocabulary_size)
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)

def masked_acc(y_true, y_pred):
    """
    Calculate the masked accuracy for each item in the batch.
    y_true.shape: (batch, t)
    y_pred.shape: (batch, t, vocabulary_size)
    """
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)