import pytest


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "slow: mark test to skip on automated tests"
    )


@pytest.fixture(scope="module", params=range(5))
def test_number(request):
    return request.param


images = {
    0: [
        [0,1],
        [0,1],
    ],
    1: [
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
    ],
    2: [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
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
        [0, 1, 0, 1, 0, 0, 2, 3, 2, 0],
        [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
}

