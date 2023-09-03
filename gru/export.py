import tensorflow as tf

"""
If you want to export this model you'll need to wrap the translate 
method in a tf.function. That implementation will get the job done.
"""

class Export(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
  def translate(self, inputs):
    return self.model.translate(inputs)