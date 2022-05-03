
from absl import flags
from absl import app
import sys
import os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import funnel_cake.gnn as gnn


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'data_directory',
    '/media/ian/Data/coarse_sim_dataset_hertz',
    'Directory which contains the train and test datasets.')
flags.DEFINE_integer(
    'time_index',
    40,
    'The time index of the target mobilities.')
flags.DEFINE_integer(
    'max_files_to_load',
    15,
    'The maximum number of files to load from the train and test datasets.')
flags.DEFINE_string(
    'checkpoint_path',
    "/media/ian/Data/tf_checkpoints",
    'Path used to store a checkpoint of the best model.')
flags.DEFINE_bool(
    "recompute_graphs",
    False,
    "Recompute graph stuctures from GSD files, even if data is already cached.")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    file_pattern = os.path.join(
        FLAGS.data_directory, 'constVolume0.95_N10000_seed10*_strainStep0.0001/traj_run2_maxStrain0.014.gsd')
    gnn.train_model_rev(
        file_pattern=file_pattern,
        test_frac=.25,
        max_files_to_load=FLAGS.max_files_to_load,
        time_index=FLAGS.time_index,
        checkpoint_path=FLAGS.checkpoint_path,
        recompute_graphs=FLAGS.recompute_graphs)


if __name__ == '__main__':
    app.run(main)
