import tensorflow._api.v2.compat.v1 as tf
import shutil

from src.save_and_load import simple_model


def do_save_model(export_dir, sess, ia, ib, out):
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.tag_constants.TRAINING],
                                         signature_def_map=
                                         {'default':
                                             tf.saved_model.signature_def_utils.build_signature_def(
                                                 inputs={
                                                     'ia': tf.saved_model.utils.build_tensor_info(ia),
                                                     'ib': tf.saved_model.utils.build_tensor_info(ib)
                                                 },
                                                 outputs={
                                                     'output': tf.saved_model.utils.build_tensor_info(out)
                                                 },
                                                 method_name='operation_add'
                                             )
                                         },
                                         )
    builder.save()


def write_graph(export_dir, graph):
    writer = tf.summary.FileWriter(export_dir, graph=graph)
    writer.flush()
    writer.close()


def clean_directories(export_dir):
    shutil.rmtree(export_dir, ignore_errors=True)


def create_base_model_and_save(export_dir):
    return do_create_model_and_save(export_dir, model_maker=simple_model.make_base_model)


def create_target_model_and_save(export_dir):
    return do_create_model_and_save(export_dir, model_maker=simple_model.make_target_model)


def create_dyn_emb_model_and_save(export_dir):
    return do_create_model_and_save(export_dir, model_maker=simple_model.make_model_dyn_emb)


def do_create_model_and_save(export_dir, model_maker):
    clean_directories(export_dir)

    with tf.Graph().as_default() as g:
        ia, ib, out = model_maker()

        with tf.Session(graph=g) as sess:
            tf.global_variables_initializer().run()

            do_save_model(export_dir, sess, ia, ib, out)
            write_graph(export_dir, sess.graph)
            saver = tf.train.Saver()
            saver.save(sess, export_dir + '/ckpt')


def show_op_names():
    for c in tf.get_default_graph().get_operations():
        if 'save' not in c.name:
            print(c.name)
