"""A graph neural network based model originally created to predict particle
mobilities.

Code reuse from 
https://github.com/deepmind/deepmind-research/tree/master/glassy_dynamics
"""

import functools
from operator import is_
import sys
from typing import Any, Dict, Text, List, Sequence, Tuple, Optional
from numba import njit
from numpy.linalg import inv

import matplotlib.pyplot as plt

from graph_nets import graphs
from graph_nets import modules as gn_modules
from graph_nets import utils_tf

import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import gsd.hoomd
import numpy as np
import os

from absl import logging
import collections
from collections import defaultdict
import enum
import freud

import pickle

# so the program doesn't crash my poor little computer
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])


# TODO need to write instructions on how to build python virt env


LossCollection = collections.namedtuple('LossCollection',
                                        'l1_loss, l2_loss, correlation')


class GraphBasedModel(snt.Module):
    """Graph based model which predicts particle mobilities from their positions.
    This network encodes the nodes and edges of the input graph independently,
    and then performs message-passing on this graph, updating its edges based
    on their associated nodes, then updating the nodes based on the input
    nodes' features and their associated updated edge features.
    This update is repeated several times.
    Afterwards the resulting node embeddings are decoded to predict the
    particle mobility.
    """

    def __init__(self,
                 n_recurrences: int,
                 mlp_sizes: Tuple[int, int],
                 mlp_kwargs: Optional[Dict[Text, Any]] = None,
                 final_layer_size: int = 1,
                 name='Graph'):
        """Creates a new GraphBasedModel object.
        Args:
            n_recurrences: the number of message passing steps in the graph
            network.
            mlp_sizes: the number of neurons in each layer of the MLP.
            mlp_kwargs: additional keyword aguments passed to the MLP.
            name: the name of the Sonnet module.
        """
        super(GraphBasedModel, self).__init__(name=name)
        self._n_recurrences = n_recurrences

        if mlp_kwargs is None:
            mlp_kwargs = {}

        model_fn = functools.partial(
            snt.nets.MLP,
            output_sizes=mlp_sizes,
            activate_final=True,
            **mlp_kwargs)

        final_model_fn = functools.partial(
            snt.nets.MLP,
            output_sizes=mlp_sizes + (final_layer_size,),
            activate_final=False,
            **mlp_kwargs)

        self._encoder = gn_modules.GraphIndependent(
            node_model_fn=model_fn,
            edge_model_fn=model_fn,
            name="decode")

        if self._n_recurrences > 0:
            self._propagation_network = gn_modules.GraphNetwork(
                node_model_fn=model_fn,
                edge_model_fn=model_fn,
                # We do not use globals, just pass the identity function.
                global_model_fn=lambda: lambda x: x,
                reducer=tf.math.unsorted_segment_mean,
                edge_block_opt=dict(use_globals=False),
                node_block_opt=dict(use_globals=False),
                global_block_opt=dict(use_globals=False),
                name="propagate")

        self._decoder_node = gn_modules.GraphIndependent(
            node_model_fn=final_model_fn,
            name="decode_nodes")

    def __call__(self, graphs_tuple: graphs.GraphsTuple, is_training=True) -> tf.Tensor:
        """Connects the model into the tensorflow graph.
        Args:
            graphs_tuple: input graph tensor as defined in `graphs_tuple.graphs`.
            is_training: NOT IMPLEMENTED, I'm a bit confused at how to do dropout
                layers in the Sonnet framework, but it should be possibly.
                In the case of a drop out layer, this parameter would change
                the computation based upon the training or test set
        Returns:
            tensor with shape [n_particles] containing the predicted particle
            mobilities.
        """
        encoded = self._encoder(graphs_tuple)
        outputs = encoded

        for _ in range(self._n_recurrences):
            # Adds skip connections by concatenating the encoding with .
            inputs = utils_tf.concat([outputs, encoded], axis=-1)
            outputs = self._propagation_network(inputs)

        inputs = utils_tf.concat([outputs, encoded], axis=-1)  # this may not be a necessary skip connection
        # outputs_edge = self._decoder_edge(outputs)
        decoded = self._decoder_node(inputs)
        return tf.squeeze(decoded.nodes, axis=-1)


class ParticleType(enum.IntEnum):
    """The simulation contains two particle types, identified as type A and B.
    The dataset encodes the particle type in an integer.
        - 0 corresponds to particle type A.
        - 1 corresponds to particle type B.
    """
    A = 0
    B = 1


@njit
def get_d2min(b0, b):
    """ Calculates D2min (and related quantities) for a set of bonds
    Args
        b0: initial bond lengths
        b: final bond lengths
    """
    dimension = b0.shape[1]
    V = b0.transpose().dot(b0)
    W = b0.transpose().dot(b)
    J = inv(V).dot(W)
    non_affine = b0.dot(J) - b
    d2min = np.sum(np.square(non_affine))
    eta = 0.5 * (J * J.transpose() - np.eye(dimension))
    eta_m = 1.0/np.double(dimension) * np.trace(eta)
    tmp = eta - eta_m * np.eye(dimension)
    eta_s = np.sqrt(0.5*np.trace(tmp*tmp))
    return (d2min, J, eta_s)


def get_log_d2min_from_pos_box(pos0, pos, nlist, box, box2):
    """
        Fetch 2D D2min 
    """

    dimension = 2
    d2mins = np.zeros((len(pos0)))

    for i in np.arange(len(pos)):
        neighbors = nlist[i]
        bonds0 = np.ascontiguousarray(
            box.wrap(
                np.array([pos0[j] - pos0[i] for j in neighbors])
            )[:, :dimension])
        bonds = np.ascontiguousarray(
            box2.wrap(
                np.array([pos[j] - pos[i] for j in neighbors]))[:, :dimension])
        d2min, _, _ = get_d2min(bonds0, bonds)
        d2mins[i] = d2min

    return d2mins


def make_graph_from_snapshots(
        snapshot: gsd.hoomd.Snapshot,
        later_snap: gsd.hoomd.Snapshot,
        edge_threshold: float = 2.0) -> Tuple[graphs.GraphsTuple, tf.Tensor]:
    """
        Returns graph representation of HOOMD snapshot
    """

    dim = snapshot.configuration.dimensions
    positions = snapshot.particles.position
    #early_positions = early_snap.particles.position
    later_positions = later_snap.particles.position

    nlist_query = freud.locality.AABBQuery.from_system(snapshot)
    nlist = nlist_query.query(
        positions,
        {'r_max': edge_threshold, 'exclude_ii': False}).toNeighborList()

    box = freud.box.Box(*snapshot.configuration.box, dim == 2)
    box2 = freud.box.Box(*later_snap.configuration.box, dim == 2)

    senders = nlist.query_point_indices[:]
    receivers = nlist.point_indices[:]
    edges = defaultdict(list)
    for i, j in zip(
            senders,
            receivers):
        edges[i].append(j)
    targets = get_log_d2min_from_pos_box(
        positions,
        later_positions,
        edges,
        box,
        box2).astype(np.float32)

    idx = 0
    edges = np.zeros((len(senders), dim))
    for i, j in zip(senders, receivers):
        edges[i] = box.wrap(positions[i] - positions[j])[:dim]
        idx += 1

    nodes = snapshot.particles.typeid[:, tf.newaxis]

    return graphs.GraphsTuple(
        nodes=tf.cast(nodes, tf.float32),
        n_node=tf.reshape(tf.shape(nodes)[0], [1]),
        edges=tf.cast(edges, tf.float32),
        n_edge=tf.reshape(tf.shape(edges)[0], [1]),
        globals=tf.zeros((1, 1), dtype=tf.float32),
        receivers=tf.cast(receivers, tf.int32),
        senders=tf.cast(senders, tf.int32)), tf.cast(targets, tf.float32), tf.cast(snapshot.particles.typeid, tf.int32)


def apply_random_rotation(graph: graphs.GraphsTuple) -> graphs.GraphsTuple:
    """Returns randomly rotated graph representation.
    The rotation is an element of O(2) with rotation angles multiple of pi/2.
    This function assumes that the relative particle distances are stored in
    the edge features.

    Args:
        graph: The graphs tuple as defined in `graph_nets.graphs`.
    """
    # Transposes edge features, so that the axes are in the first dimension.
    # Outputs a tensor of shape [2, n_particles].
    xy = tf.transpose(graph.edges)
    # Random pi/2 rotation(s)
    permutation = tf.random.shuffle(tf.constant([0, 1], dtype=tf.int32))
    xy = tf.gather(xy, permutation)
    # Random reflections.
    symmetry = tf.random.uniform([2], minval=0, maxval=2, dtype=tf.int32)
    symmetry = 1 - 2 * tf.cast(tf.reshape(symmetry, [2, 1]), tf.float32)
    xyz = xy * symmetry
    edges = tf.transpose(xyz)
    return graph.replace(edges=edges)


def load_data_gsd(
        file_pattern: Text,
        time_index: int,
        edge_threshold: float = 2.0,
        max_files_to_load: Optional[int] = None,
        recompute_graphs: bool = False) -> List[Tuple[graphs.GraphsTuple, tf.Tensor]]:
    """

    """
    filenames = tf.io.gfile.glob(file_pattern)
    print(len(filenames), "files match the wildcard")
    if max_files_to_load:
        filenames = filenames[:max_files_to_load]

    static_structures = []
    targets = []
    typesl = []
    print("Num files:", len(filenames))
    for filename in filenames:
        maybe_pickle_file = filename.replace(".gsd", "_graph.pickle")
        assert maybe_pickle_file != filename
        try:
            if recompute_graphs:
                raise Exception
            with open(maybe_pickle_file, 'rb') as f:
                pickle_data = pickle.load(f)
                static_structures.extend(pickle_data["graph"])
                targets.extend(pickle_data["target"])
                typesl.extend(pickle_data["types"])
        except:
            with gsd.hoomd.open(name=filename, mode="rb") as data:
                num_frames = len(data)
                g = []
                ta = []
                t = []
                for frame in np.arange(0, num_frames-80, time_index, dtype=int):  # TODO the 39 is specific to my data
                    frame = int(frame)
                    graph, target, types = make_graph_from_snapshots(
                        data[frame],
                        data[frame+time_index],
                        edge_threshold=edge_threshold)
                    g.append(graph)
                    ta.append(target)
                    t.append(types)
                with open(maybe_pickle_file, "wb") as f:
                    pickle.dump({"graph":g, "target":ta, "types":t}, f)
                static_structures.extend(g)
                targets.extend(ta)
                typesl.extend(t)
    return static_structures, targets, typesl


def get_loss_ops(
        prediction: tf.Tensor,
        target: tf.Tensor,
        types: tf.Tensor,
        batch_mask: tf.Tensor,
) -> LossCollection:
    """Returns L1/L2 loss and correlation for type A particles.
    Args:
        prediction: tensor with shape [n_particles] containing the predicted
            particle mobilities.
        target: tensor with shape [n_particles] containing the true particle
            mobilities.
        types: tensor with shape [n_particles] containing the particle types.
    """
    # Considers only type A particles.
    mask = tf.equal(types, ParticleType.A)
    mask = tf.logical_and(mask, batch_mask)
    prediction = tf.boolean_mask(prediction, mask)
    target = tf.boolean_mask(target, mask)
    return LossCollection(
        l1_loss=tf.reduce_mean(tf.abs(prediction - target)),
        l2_loss=tf.reduce_mean((prediction - target)**2),
        correlation=tf.squeeze(tfp.stats.correlation(
            prediction[:, tf.newaxis], target[:, tf.newaxis])))


def _log_stats_and_return_mean_correlation(
        label: Text,
        stats: Sequence[LossCollection]) -> float:
    """Logs performance statistics and returns mean correlation.
    Args:
        label: label printed before the combined statistics e.g. train or test.
        stats: statistics calculated for each batch in a dataset.
    Returns:
        mean correlation
    """
    for key in LossCollection._fields:
        values = [getattr(s, key) for s in stats]
        mean = np.mean(values)
        std = np.std(values)
        logging.info('%s: %s: %.4f +/- %.4f', label, key, mean, std)
    return np.mean([s.correlation for s in stats])


def train_model(file_pattern: Text,
                test_frac: float,
                max_files_to_load: Optional[int] = None,
                n_epochs: int = 10_000,
                time_index: int = 40,
                augment_data_using_rotations: bool = False,
                learning_rate: float = 1e-3,
                grad_clip: Optional[float] = 1.0,
                n_recurrences: int = 2,
                mlp_sizes: Tuple[int, int] = (32, 32),
                mlp_kwargs: Optional[Dict[Text, Any]] = None,
                edge_threshold: float = 2.0,
                measurement_store_interval: int = 10,
                checkpoint_path: Optional[Text] = None,
                mini_batch: Optional[int] = 64,
                recompute_graphs: bool = False) -> float:  # pytype: disable=annotation-type-mismatch
    """Trains GraphModel using tensorflow.
    Args:
        train_file_pattern: pattern matching the files with the training data.
        test_frac: fraction of data to use for testing.
        max_files_to_load: the maximum number of train and test files to load.
        If None, all files will be loaded.
        n_epochs: the number of passes through the training dataset (epochs).
        time_index: the time index (0-9) of the target mobilities.
        augment_data_using_rotations: data is augemented by using random rotations.
        learning_rate: the learning rate used by the optimizer.
        grad_clip: all gradients are clipped to the given value.
        n_recurrences: the number of message passing steps in the graphnet.
        mlp_sizes: the number of neurons in each layer of the MLP.
        mlp_kwargs: additional keyword aguments passed to the MLP.
        edge_threshold: particles at distance less than threshold are connected by
        an edge.
        measurement_store_interval: number of steps between storing objective values
        (loss and correlation).
        checkpoint_path: path used to store the checkpoint with the highest
        correlation on the test set.
    Returns:
        Correlation on the test dataset of best model encountered during training.
    """

    # Loads train and test dataset.
    dataset_kwargs = dict(
        time_index=time_index,
        max_files_to_load=max_files_to_load,
        edge_threshold=edge_threshold,
        recompute_graphs=recompute_graphs)
    my_data, my_targets, my_types = load_data_gsd(file_pattern, **dataset_kwargs)

    save_prefix = os.path.join(checkpoint_path, "my_model/test3")

    # this is a bit crude, but just shift and rescale the targets
    mean_targets = np.mean(np.log(my_targets))
    std_targets = np.std(np.log(my_targets))
    my_targets = [(np.log(t)-mean_targets)/std_targets for t in my_targets]
    print(mean_targets, std_targets)
    plt.hist(np.array(my_targets[0]), bins=25)

    # define optimizer, model, and checkpoint
    optimizer = tf.keras.optimizers.Adam(learning_rate, clipnorm=grad_clip)
    model = GraphBasedModel(n_recurrences, mlp_sizes, mlp_kwargs)
    checkpoint = tf.train.Checkpoint(module=model)

    # uncompiled training function
    # we have to do some work below to compile it onto our target device
    def training_step(graph, targets, types, batch_mask, mini_batch_size):
        losses = []
        with tf.GradientTape() as tape:
            prediction = model(graph, is_training=True)
            loss = get_loss_ops(prediction, targets, types)
            losses.append(loss)
            gradients = tape.gradient(
                tf.stack([loss.l2_loss]), model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
        return losses

    # split up training and test sets
    # again, a bit crude because we aren't doing a shuffle
    split = int(len(my_data)*(1-test_frac))
    training_graphs = my_data[:split]
    training_targets = tf.stack(my_targets[:split])
    training_types = tf.stack(my_types[:split])
    test_graphs = my_data[split:]
    test_targets = my_targets[split:]
    test_types = tf.stack(my_types[:split])

    latest = tf.train.latest_checkpoint(save_prefix)
    # if latest is not None:
    #     checkpoint.restore(latest)

    # defines the shapes of our input types
    # this is necessary because the graphs have variable shape and tf.function
    # needs to be made aware of this
    input_signature = [
        utils_tf.specs_from_graphs_tuple(training_graphs[0]),
        tf.TensorSpec(tf.shape(training_targets[0])),
        tf.TensorSpec(tf.shape(training_types[0]), dtype=tf.int32),
        tf.TensorSpec(tf.shape(training_types[0]), dtype=tf.bool),
        tf.int32
    ]
    compiled_training_step = tf.function(training_step, input_signature=input_signature)

    best_so_far = -1

    for i in range(n_epochs):
        train_stats = []
        test_stats = []
        perm = tf.random.shuffle(tf.range(split))
        for j in range(split):
            batch = tf.random.shuffle(tf.range(split))
            if augment_data_using_rotations:
                train_loss = compiled_training_step(
                    apply_random_rotation(training_graphs[perm[j]]),
                    training_targets[perm[j]],
                    training_types[perm[j]])
            else:
                train_loss = compiled_training_step(
                    training_graphs[perm[j]],
                    training_targets[perm[j]],
                    training_types[perm[j]])
            train_stats.extend(train_loss)

        if (i+1) % measurement_store_interval == 0:
            # Evaluates model on test dataset.

            # we could have wrapped this around a tf.function, but eh
            # performance penalty is probably mute
            for k in range(len(test_graphs)):
                pred = model(test_graphs[k])
                y = test_targets[k]
                test_stats.append(get_loss_ops(pred, y, test_types[k]))

            # Outputs performance statistics on training and test dataset.
            _log_stats_and_return_mean_correlation('Train', train_stats)
            correlation = _log_stats_and_return_mean_correlation(
                'Test', test_stats)

            # Updates best model based on the observed correlation on the test
            # dataset.
            if correlation > best_so_far:
                best_so_far = correlation
                if checkpoint_path:
                    checkpoint.save(save_prefix)

    # checkpoint.save(save_prefix)
    return best_so_far


def apply_model(file_pattern: Text,
                max_files_to_load: Optional[int] = None,
                time_index: int = 40,
                n_recurrences: int = 4,
                mlp_sizes: Tuple[int, int] = (64, 64),
                mlp_kwargs: Optional[Dict[Text, Any]] = None,
                edge_threshold: float = 1.5,
                checkpoint_path: Optional[Text] = None) -> float:  # pytype: disable=annotation-type-mismatch
    """
        TODO this is not the best way of applying a finalized tf model, but ...
        Will be used to do some initial tests of the graph module 
    """
    if mlp_kwargs is None:
        mlp_kwargs = dict(w_init=tf.keras.initializers.VarianceScaling(scale=.5),
                          b_init=tf.keras.initializers.VarianceScaling(scale=.5))
    # Loads train and test dataset.
    dataset_kwargs = dict(
        time_index=time_index,
        max_files_to_load=max_files_to_load,
        edge_threshold=edge_threshold)
    my_data, my_targets, my_types = load_data_gsd(file_pattern, **dataset_kwargs)

    model = GraphBasedModel(n_recurrences, mlp_sizes, mlp_kwargs)
    checkpoint = tf.train.Checkpoint(module=model)

    latest = tf.train.latest_checkpoint(checkpoint_path+"/my_model")
    if latest is not None:
        print("found model")
        checkpoint.restore(latest)

    # Evaluates model on data

    for k in range(len(my_data)):
        pred = model(my_data[k])
        y = my_targets[k]

    return pred, y, my_types
