"""Shared pytest configuration for the udg test suite."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: long-running regression test (deselect with -m 'not slow')",
    )
