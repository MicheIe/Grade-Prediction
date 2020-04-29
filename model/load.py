from keras.models import model_from_json
import tensorflow as tf

def init():
    json_file = open('/Users/apple/Documents/model/model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('/Users/apple/Documents/model/model.h5')
    print('Loaded Model From Disk')
    loaded_model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_absolute_error'])
    graph = tf.get_default_graph()
    return loaded_model, graph



