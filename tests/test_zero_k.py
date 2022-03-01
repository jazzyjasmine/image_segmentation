import numpy as np

import tests.conftest
import segment
import utils


def test_zero_k(test_number):
    image = np.array(tests.conftest.images[test_number])
    expected = np.array(expected_dict[test_number])
    result = segment.segment_image(0, image)
    return utils.assert_segments_equal(result, expected)


expected_dict = {
    0: [
        [0, 1],
        [0, 1],
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
        [0, 1, 2, 1, 0],
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
        [0, 1, 4, 1, 0, 0, 2, 3, 2, 0],
        [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
}
