import sys
import threading
import pytest
import numpy
import pyca
from conftest import all_pvs, waveform_pvs, ctrl_keys, calc_new_value


class ConnectCallback(object):
    def __init__(self, name):
        self.name = name
        self.connected = False
        self.cev = threading.Event()
        self.dcev = threading.Event()
        self.lock = threading.RLock()

    def wait(self, timeout=None):
        return self.cev.wait(timeout=timeout)

    def wait_dc(self, timeout=None):
        return self.dcev.wait(timeout=timeout)

    def __call__(self, is_connected):
        with self.lock:
            self.connected = is_connected
            if self.connected:
                self.cev.set()
                self.dcev.clear()
            else:
                self.dcev.set()
                self.cev.clear()


class GetCallback(object):
    def __init__(self, name):
        self.name = name
        self.gev = threading.Event()

    def wait(self, timeout=None):
        return self.gev.wait(timeout=timeout)

    def reset(self):
        self.gev.clear()

    def __call__(self, exception=None):
        if exception is None:
            self.gev.set()


class MonitorCallback(object):
    def __init__(self, name):
        self.name = name
        self.mev = threading.Event()

    def wait(self, timeout=None):
        return self.mev.wait(timeout=timeout)

    def reset(self):
        self.mev.clear()

    def __call__(self, exception=None):
        if exception is None:
            self.mev.set()


def setup_pv(name, connect=True):
    pv = pyca.capv(name)
    pv.connect_cb = ConnectCallback(name)
    pv.getevt_cb = GetCallback(name)
    pv.monitor_cb = MonitorCallback(name)
    if connect:
        pv.create_channel()
        assert pv.connect_cb.wait(timeout=1)
    return pv


ReadonlyError = AttributeError if sys.version_info.major >= 3 else TypeError


@pytest.fixture(params=all_pvs)
def any_pv(server, request):
    pv = setup_pv(request.param)
    yield pv
    try:
        pv.clear_channel()
    except pyca.pyexc:
        pass

@pytest.fixture(params=waveform_pvs)
def waveform_pv(server, request):
    pv = setup_pv(request.param)
    yield pv
    try:
        pv.clear_channel()
    except pyca.pyexc:
        pass


def test_server_start(server):
    pass

def test_name_readonly():
    pv = setup_pv('test', False)
    with pytest.raises(ReadonlyError):
        pv.name = "new name"

def test_data_readonly():
    pv = setup_pv('test', False)
    pv.data['value'] = "value"
    with pytest.raises(ReadonlyError):
        pv.data = []

def test_name():
    pvname = 'testname'
    pv = setup_pv(pvname, False)
    assert pv.name == pvname


@pytest.mark.timeout(10)
def test_create_and_clear_channel(any_pv):
    any_pv.clear_channel()
    pyca.flush_io()
    assert not any_pv.connect_cb.wait_dc(timeout=1)
    with pytest.raises(pyca.pyexc):
        any_pv.get_data(False, -1.0)

@pytest.mark.timeout(10)
def test_get_time_data(any_pv):
    any_pv.get_data(False, -1.0)
    pyca.flush_io()
    assert any_pv.getevt_cb.wait(timeout=1)
    assert 'value' in any_pv.data
    assert any_pv.data['value'] is not None

@pytest.mark.timeout(10)
def test_get_ctrl_data(any_pv):
    any_pv.get_data(True, -1.0)
    pyca.flush_io()
    assert any_pv.getevt_cb.wait(timeout=1)
    assert 'value' in any_pv.data
    assert any_pv.data['value'] is not None
    for key in ctrl_keys[any_pv.name]:
        assert key in any_pv.data

@pytest.mark.timeout(10)
def test_put_get(any_pv):
    any_pv.get_data(False, -1.0)
    pyca.flush_io()
    assert any_pv.getevt_cb.wait(timeout=1)
    new_value = calc_new_value(any_pv.name, any_pv.data['value'])
    any_pv.put_data(new_value, 1.0)
    any_pv.getevt_cb.reset()
    any_pv.get_data(False, -1.0)
    pyca.flush_io()
    assert any_pv.getevt_cb.wait(timeout=1)
    recv_value = any_pv.data['value']
    assert recv_value == new_value


@pytest.mark.timeout(10)
def test_subscribe(any_pv):
    any_pv.get_data(False, -1.0)
    pyca.flush_io()
    assert any_pv.getevt_cb.wait(timeout=1)
    new_value = calc_new_value(any_pv.name, any_pv.data['value'])
    any_pv.subscribe_channel(pyca.DBE_VALUE | pyca.DBE_LOG | pyca.DBE_ALARM, False)
    pyca.flush_io()
    # subscription start monitor
    assert any_pv.monitor_cb.wait(timeout=1)
    any_pv.monitor_cb.reset()
    any_pv.put_data(new_value, 1.0)
    assert any_pv.monitor_cb.wait(timeout=1)
    assert any_pv.data['value'] == new_value


@pytest.mark.timeout(10)
def test_misc(any_pv):
    assert isinstance(any_pv.host(), str)
    assert isinstance(any_pv.state(), int)
    assert isinstance(any_pv.count(), int)
    assert isinstance(any_pv.type(), str)
    assert isinstance(any_pv.rwaccess(), int)


@pytest.mark.timeout(10)
def test_waveform_tuple(waveform_pv):
    waveform_pv.use_numpy = False
    waveform_pv.get_data(False, -1.0)
    pyca.flush_io()
    assert waveform_pv.getevt_cb.wait(timeout=1)
    val = waveform_pv.data['value']
    assert isinstance(val, list)
    assert len(val) == waveform_pv.count()

@pytest.mark.timeout(10)
def test_waveform_numpy(waveform_pv):
    waveform_pv.use_numpy = True
    waveform_pv.get_data(False, -1.0)
    pyca.flush_io()
    assert waveform_pv.getevt_cb.wait(timeout=1)
    val = waveform_pv.data['value']
    assert isinstance(val, numpy.ndarray)
    assert len(val) == waveform_pv.count()


@pytest.mark.timeout(10)
def test_enum_strings(server):
    pvname = 'PYCA:TEST:ENUM'
    pv = setup_pv(pvname)
    pv.get_enum_strings(-1.0)
    pyca.flush_io()
    assert pv.getevt_cb.wait(timeout=1)
    assert pv.data['enum_set'] == ('A', 'B')
    pv.clear_channel()


@pytest.mark.timeout(10)
def test_threads(server):
    def some_thread_thing(pvname):
        pyca.attach_context()
        pv = setup_pv(pvname)
        pv.get_data(False, -1.0)
        pyca.flush_io()
        assert pv.getevt_cb.wait(timeout=1)
        assert 'value' in pv.data
        pv.clear_channel()

    threads = [ threading.Thread(target=some_thread_thing, args=(x,)) for x in all_pvs ]
    for x in threads:
        x.start()
    for x in threads:
        x.join()
