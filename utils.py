import numpy as np
from PIL import Image

from segment import segment_image


def load_image_as_grayscale_array(filename: str) -> np.ndarray:
    """
    load an image file and return it as 2D np.ndarray of floats ranging from 0 to 255
    where each float corresponds to a pixel in grayscale image

    :param filename: path to image file
    :return: a 2D np.ndarray representing a grayscale image
    """
    im = Image.open(filename)
    grayscale = im.convert("L")
    arr = np.asarray(grayscale).astype(int)
    return arr


def save_array_as_image(image: np.ndarray, filename: str):
    """
    save a grayscale image to disk

    :param image: a 2D np.ndarray representing an image
    :param filename: path to save image
    """
    im = Image.fromarray(image, "L")
    im.save(filename, "PNG")


def normalize_segmented_image(image: np.ndarray, segment_jump: int = 31):
    """
    convert a 2D np.ndarray of segment labels to an np.ndarray ranging from 0 to 255
    which represents a grayscale image

    :param image: a 2D np.ndarray representing a segmented image
    :param segment_jump: how far apart two segments should be when creating them
    :return: a 2D np.ndarray representing a grayscale image
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    segments = {}
    next_segment = -1
    out = np.zeros(image.size)
    for idx, label in enumerate(image.flatten()):
        if label not in segments:
            next_segment = (next_segment + segment_jump) % 256  # give nearby labels diff grayscale values
        out[idx] = segments.setdefault(label, next_segment)
    out = (out - out.min()) / (out.max() - out.min()) * 255
    return out.reshape(image.shape).astype(np.uint8)


def load_segment_save_image(k: int, in_filename: str, out_filename: str, segment_jump: int = 31):
    """
    load an image in grayscale, segment it, and save it to disk

    :param k: parameter into image segmentation algorithm,
    affects number of segments created
    :param in_filename: path to image file
    :param out_filename: path to save result
    :param segment_jump: how far apart two segments should be when creating them
    """
    image = load_image_as_grayscale_array(in_filename)
    image = segment_image(k, image)
    image = normalize_segmented_image(image, segment_jump)
    save_array_as_image(image, out_filename)


def assert_segments_equal(image, expected):
    """
    helper to assert that two segments are equal, raises AssertionError if they are not labeled the same
    :param image: a 2D np.ndarray representing a segmented image
    :param expected: a 2D np.ndarray representing a segmented image
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    assert image.shape == expected.shape
    label_idx = 0
    # translate both image and expected labels to a new set of labels
    image_labels = {}
    expected_labels = {}
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] not in image_labels and expected[i][j] not in expected_labels:
                # first time seeing this label, store both the label and first occurence of this label
                image_labels[image[i][j]] = label_idx, (i,j)
                expected_labels[expected[i][j]] = label_idx, (i,j)
                label_idx += 1
            elif image[i][j] not in image_labels:
                # this branch will lead to an assertion error
                image_labels[image[i][j]] = label_idx, (i,j)
                label_idx += 1
            elif expected[i][j] not in expected_labels:
                # this branch will lead to an assertion error
                expected_labels[expected[i][j]] = label_idx, (i,j)
                label_idx += 1
            image_label, image_first = image_labels[image[i][j]]
            expected_label, expected_first = expected_labels[expected[i][j]]

            if expected_first == (i,j):
                err_msg = f"Incorrect segment for pixel ({i},{j}):\n" \
                    f"Pixel ({i}, {j}) segment: {image[i][j]}\n" \
                    f"Expected new segment label, but matches pixel {image_first[0], image_first[1]}'s segment: " \
                    f"{image[image_first[0]][image_first[1]]}\n"
            else:
                err_msg = f"Incorrect segment for pixel ({i},{j}):\n" \
                f"Pixel ({i}, {j}) segment: {image[i][j]}\n" \
                f"Expected segment to match pixel {expected_first[0], expected_first[1]}'s segment: " \
                f"{image[expected_first[0]][expected_first[1]]}\n"
            assert image_label == expected_label, err_msg
