import tensorflow as tf

# Protocol Buffer
int_f = tf.train.Feature(
    int64_list=tf.train.Int64List(value=[1, 2]))
#print(repr(int_f) + '\n')

float_f = tf.train.Feature(
    float_list=tf.train.FloatList(value=[-8.2, 5]))
#print(repr(float_f) + '\n')

bytes_f = tf.train.Feature(
    bytes_list=tf.train.BytesList(value=[b'\xff\xcc', b'\xac']))
#print(repr(bytes_f) + '\n')

str_f = tf.train.Feature(
    bytes_list=tf.train.BytesList(value=['joe'.encode()]))
#print(repr(str_f) + '\n)

f_dict = {
    'int_vals': int_f,
    'float_vals': float_f,
    'bytes_vals': bytes_f,
    'str_vals': str_f
}

def dict_to_example(data_dict, config):
    feature_dict = {}
    for feature_name, value in data_dict.items():
        feature_config = config[feature_name]
        shape = feature_config['shape']
        if shape == () or shape == []:
            value = [value]
        value_type = feature_config['type']
        if value_type == 'int':
            feature_dict[feature_name] = make_int_feature(value)
        elif value_type == 'float':
            feature_dict[feature_name] = make_float_feature(value)
        elif value_type == 'string' or value_type == 'bytes':
            feature_dict[feature_name] = make_bytes_feature(
                value, value_type)
    features = tf.train.Features(feature=feature_dict)
    return tf.train.Example(features=features)

# Serialization ~ convert to byte string which can be written
# to a file
ex = tf.train.Example(features=tf.train.Features(feature=f_dict))
print(repr(ex))

# How to serialize an object
ser_ex = ex.SerializeToString()
print(ser_ex)

# Writing to data files
writer = tf.python_io.TFRecordWriter('out.tfrecords')
writer.write(ser_ex)
writer.close()