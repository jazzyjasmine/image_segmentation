import numpy as np

import tests.conftest
import segment
import utils


def test_positive_k(test_number):
    image = np.array(tests.conftest.images[test_number])
    expected = np.array(expected_dict[test_number])
    k = 3 if test_number == 0 or test_number == 1 else 9
    result = segment.segment_image(k, image)
    return utils.assert_segments_equal(result, expected)


expected_dict = {
    0: [
        [0, 0],
        [0, 0],
    ],
    1: [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ],
    2: [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],  # center is now the same segment because of k
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ],
    3: [
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ],
    4: [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
        [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
        [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
}
