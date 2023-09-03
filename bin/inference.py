#%%
import tensorflow as tf
import tensorflow_text # without this import the model 
                       # loading will not work

# %% 
inputs = [
    'Hace mucho frio aqui.', # "It's really cold here."
    'Esta es mi vida.', # "This is my life."
    'Su cuarto es un desastre.', # "His room is a mess"
    'yo como papa hoy.', # "i eat potato today."
    'El puede hablar muy bien alemana.'  # "he can speak very good german."
]

#%%
%%time
reloaded = tf.saved_model.load('translator')
_ = reloaded.translate(tf.constant(inputs)) #warmup

#%%
%%time
result = reloaded.translate(tf.constant(inputs))

for i in range(len(result)):
    print(result[i].numpy().decode())

print()
# %%
