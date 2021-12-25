import numpy as np

import export_paths
import simple_model
import inner_export_utils
import tensorflow._api.v2.compat.v1 as tf
import numpy as np

# import tensorflow_recommenders_addons.dynamic_embedding as de

# tf.disable_eager_execution()

export_dir = export_paths.dyn_emb_dir

if __name__ == '__main__':
    inner_export_utils.create_dyn_emb_model_and_save(export_dir)
