import export_paths
import inner_export_utils

export_dir = export_paths.base_dir

if __name__ == '__main__':
    inner_export_utils.create_model_and_save(export_dir)
