import pytest


def pytest_collection_modifyitems(config, items):
    markexpr = (config.option.markexpr or "").strip()
    explicit_external = "external_network" in markexpr
    if explicit_external:
        return

    skip_external = pytest.mark.skip(
        reason="external_network tests run only when explicitly selected with -m external_network"
    )
    for item in items:
        if item.get_closest_marker("external_network"):
            item.add_marker(skip_external)
