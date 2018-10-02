import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.contrib.lite.python.convert_saved_model import get_tensors_from_tensor_names, set_tensor_shapes

inputs = ['Placeholder']
input_shapes = {'Placeholder': [1, 512, 512, 3]}
outputs = ['logits/Conv2D']
output_shapes = {'logits/Conv2D': [1, 512, 512, 32]}
transforms = [
    'strip_unused_nodes(type=float,shape=\"1,512,512,3\")',
    'fold_constants(ignore_errors=true)',
    'fold_batch_norms',
    'fold_old_batch_norms'
]

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('checkpoints/model.ckpt.meta')
    saver.restore(sess, "checkpoints/model.ckpt")

    # tensorboard
    # nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # file_writer = tf.summary.FileWriter(logdir='logs/my-model', graph=tf.get_default_graph())
    # file_writer.flush()
    # file_writer.close()
    input_tensors = get_tensors_from_tensor_names(sess.graph, inputs)
    output_tensors = get_tensors_from_tensor_names(sess.graph, outputs)
    set_tensor_shapes(input_tensors, input_shapes)
    set_tensor_shapes(output_tensors, output_shapes)

    optimized_graph = TransformGraph(
        sess.graph_def,
        inputs,
        outputs,
        transforms
    )

    # Output nodes
    output_node_names = [
        n.name for n in optimized_graph.node]

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        optimized_graph,
        output_node_names)

    # # Save the frozen graph
    # with open('output_graph.pb', 'wb') as f:
    #     f.write(frozen_graph_def.SerializeToString())

    # graph_def_file = 'output_graph.pb'

    converter = tf.contrib.lite.TocoConverter(frozen_graph_def, input_tensors, output_tensors)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)
