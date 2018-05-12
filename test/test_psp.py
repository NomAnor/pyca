import time
import threading
import pytest
import numpy
import psp
from conftest import all_pvs, waveform_pvs, ctrl_keys, calc_new_value


def setup_pv(name, connect=True):
    pv = psp.PV(name)
    if connect:
        pv.connect(timeout=1)
        assert pv.isconnected
    return pv


@pytest.fixture(params=all_pvs)
def any_pv(server, request):
    pv = setup_pv(request.param)
    yield pv
    if pv.isconnected:
        pv.disconnect()

@pytest.fixture(params=waveform_pvs)
def waveform_pv(server, request):
    pv = setup_pv(request.param)
    yield pv
    if pv.isconnected:
        pv.disconnect()


def test_server_start(server):
    pass


@pytest.mark.timeout(10)
def test_connect_and_disconnect(any_pv):
    assert any_pv.isconnected
    any_pv.disconnect()
    assert not any_pv.isconnected


@pytest.mark.timeout(10)
def test_get_data(any_pv):
    assert any_pv.get() is not None


@pytest.mark.timeout(10)
def test_put_get(any_pv):
    new_value = calc_new_value(any_pv.name, any_pv.get())
    any_pv.put(new_value, timeout=1.0)
    assert any_pv.get() == new_value


@pytest.mark.timeout(10)
def test_monitor(any_pv):
    new_value = calc_new_value(any_pv.name, any_pv.get())
    any_pv.monitor()
    any_pv.put(new_value)
    n = 0
    while n < 10 and any_pv.value != new_value:
        time.sleep(0.1)
        n += 1
    assert any_pv.value == new_value


@pytest.mark.timeout(10)
def test_misc(any_pv):
    assert isinstance(any_pv.host(), str)
    assert isinstance(any_pv.state(), int)
    assert isinstance(any_pv.count, int)
    assert isinstance(any_pv.type(), str)
    assert isinstance(any_pv.rwaccess(), int)


@pytest.mark.timeout(10)
def test_waveform_list(waveform_pv):
    waveform_pv.use_numpy = False
    value = waveform_pv.get()
    assert isinstance(value, list)
    assert len(value) == waveform_pv.count

@pytest.mark.timeout(10)
def test_waveform_numpy(waveform_pv):
    waveform_pv.use_numpy = True
    value = waveform_pv.get()
    assert isinstance(value, numpy.ndarray)
    assert len(value) == waveform_pv.count


@pytest.mark.timeout(10)
def test_enum_set(server):
    pvname = 'PYCA:TEST:ENUM'
    pv = setup_pv(pvname)
    assert pv.get_enum_set() == ('A', 'B')
    pv.disconnect()


@pytest.mark.timeout(10)
def test_threads(server):
    def some_thread_thing(pvname):
        psp.utils.ensure_context()
        pv = setup_pv(pvname)
        assert pv.get() is not None
        pv.disconnect()

    threads = [ threading.Thread(target=some_thread_thing, args=(x,)) for x in all_pvs ]
    for x in threads:
        x.start()
    for x in threads:
        x.join()
