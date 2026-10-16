"""The checks run offline: no judge reaches a model (the answer judge is played by code), and an async test runs on an
event loop of its own."""

import asyncio
import inspect

import pytest

from .. import session
from ..forge import tries
from .flows import scripted_judges


@pytest.fixture(autouse=True, scope="session")
def no_model():
    with scripted_judges():
        yield


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    """A lost session is taken up again at once: the checks wait for no sandbox."""
    monkeypatch.setattr(session, "LOSS_WAITS", (0,))
    monkeypatch.setattr(tries, "WAITS", (0, 0, 0))


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if not inspect.iscoroutinefunction(pyfuncitem.obj):
        return None
    arguments = {name: pyfuncitem.funcargs[name] for name in inspect.signature(pyfuncitem.obj).parameters}
    asyncio.run(pyfuncitem.obj(**arguments))
    return True
