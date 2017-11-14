import collections
import tensorflow as tf
from numpy.lib.arraypad import _validate_lengths
import numpy as np

__all__ = ['tf_pad_wrap']


_DummyArray = collections.namedtuple("_DummyArray", ["ndim"])


def _pad_wrap(arr, pad_amt, axis=-1):
    """
    Modified from numpy.lib.arraypad._pad_wrap
    """

    # Implicit booleanness to test for zero (or None) in any scalar type
    if pad_amt[0] == 0 and pad_amt[1] == 0:
        return arr

    ##########################################################################
    # Prepended region

    # Slice off a reverse indexed chunk from near edge to pad `arr` before
    start = arr.shape[axis] - pad_amt[0]
    end = arr.shape[axis]
    wrap_slice = tuple(slice(None) if i != axis else slice(start, end)
                       for (i, x) in enumerate(arr.shape))
    wrap_chunk1 = arr[wrap_slice]

    ##########################################################################
    # Appended region

    # Slice off a reverse indexed chunk from far edge to pad `arr` after
    wrap_slice = tuple(slice(None) if i != axis else slice(0, pad_amt[1])
                       for (i, x) in enumerate(arr.shape))
    wrap_chunk2 = arr[wrap_slice]

    # Concatenate `arr` with both chunks, extending along `axis`
    return tf.concat((wrap_chunk1, arr, wrap_chunk2), axis=axis)


def tf_pad_wrap(array, pad_width):
    """
    Modified from numpy.lib.arraypad.wrap
    """

    if not np.asarray(pad_width).dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.')

    pad_width = _validate_lengths(_DummyArray(array.get_shape().ndims), pad_width)

    for axis, (pad_before, pad_after) in enumerate(pad_width):
        if array.get_shape().as_list()[axis] is None and (pad_before > 0 or pad_after > 0):
            raise TypeError('`pad_width` must be zero for dimensions that are None.')

    # If we get here, use new padding method
    newmat = tf.identity(array)

    for axis, (pad_before, pad_after) in enumerate(pad_width):
        # Recursive padding along any axis where `pad_amt` is too large
        # for indexing tricks. We can only safely pad the original axis
        # length, to keep the period of the reflections consistent.
        safe_pad = newmat.get_shape().as_list()[axis]

        if safe_pad is None:
            continue

        while ((pad_before > safe_pad) or
               (pad_after > safe_pad)):
            pad_iter_b = min(safe_pad,
                             safe_pad * (pad_before // safe_pad))
            pad_iter_a = min(safe_pad, safe_pad * (pad_after // safe_pad))
            newmat = _pad_wrap(newmat, (pad_iter_b, pad_iter_a), axis)

            pad_before -= pad_iter_b
            pad_after -= pad_iter_a
            safe_pad += pad_iter_b + pad_iter_a
        newmat = _pad_wrap(newmat, (pad_before, pad_after), axis)

    return newmat


if __name__ == "__main__":
    import numpy as np
    session = tf.InteractiveSession()
    
    # Pad a singleton array
    print("Paddin a singleton")
    print("------------------\n")

    p = 1
    np_j = np.arange(1).reshape(1, 1) + 1
    np_k = np.pad(np_j, 1, "wrap")

    print("Padding:", p)
    print("Array:\n", np_j)
    print("\nNP:\n", np_k)

    tf_j = tf.placeholder(tf.float32, [1,1])
    tf_k = tf_pad_wrap(tf_j, 1)

    print("\nTF:\n", session.run(tf_k, feed_dict={tf_j: np_j}))
    print("\n\n")
    
    # Pad an array
    for p in (1,2):
        print("Simple 4x4 emaple")
        print("-----------------\n")

        np_a = np.arange(16).reshape(4, 4) + 1
        np_b = np.pad(np_a, p, "wrap")

        print("Padding:", p)
        print("Array:\n", np_a)
        print("\nNP:\n", np_b)

        tf_a = tf.placeholder(tf.float32, [4, 4])
        tf_b = tf_pad_wrap(tf_a, p)

        session = tf.InteractiveSession()
        print("\nTF:\n", session.run(tf_b, feed_dict={tf_a: np_a}))

        print("\n\n")

    # Test padding of array with unknown dim
    print("Example with tf tensor with unknown dim")
    print("---------------------------------------\n")

    np_d = np.pad(np_a, [(0, 0), (2, 2)], "wrap")

    print("Padding:", [(0, 0), (2, 2)])
    print("Array:\n", np_a)
    print("\nNP:\n", np_d)

    tf_c = tf.placeholder(tf.float32, [None, 4])
    tf_d = tf_pad_wrap(tf_c, [(0, 0), (2, 2)])
    print("\nTF:\n", session.run(tf_d, feed_dict={tf_c: np_a}))

    print("\n\n")

    # Test more dimensions
    print("Example with 3D tensor with unknown dim")
    print("---------------------------------------\n")

    np_x = np.arange(18).reshape(2, 3, 3) + 1
    np_y = np.pad(np_x, [(0, 0), (2, 2), (0, 0)], "wrap")

    print("Padding:", [(0, 0), (2, 2), (0, 0)])
    print("Array:\n", np_x)
    print("\nNP:\n", np_y)

    tf_x = tf.placeholder(tf.float32, [None, 3, 3])
    tf_y = tf_pad_wrap(tf_x, [(0, 0), (2, 2), (0, 0)])

    session = tf.InteractiveSession()

    print("\nTF:\n", session.run(tf_y, feed_dict={tf_x: np_x}))

    print("\n\n")
