import sys
from pcaspy import Driver, SimpleServer
from pcaspy.tools import ServerThread
import pytest

if sys.version_info.major >= 3:
    long = int

pvdb = {
    'PYCA:TEST:LONG': dict(type="int"),
    'PYCA:TEST:DOUBLE': dict(type="float"),
    'PYCA:TEST:STRING': dict(type="string"),
    'PYCA:TEST:ENUM': dict(type="enum", enums=('A', 'B')),
    'PYCA:TEST:WAVE_LONG': dict(type="int", count=10),
    'PYCA:TEST:WAVE_DOUBLE': dict(type="float", count=10)
}
ctrl_keys_string = ['status', 'severity']
ctrl_keys_enum = ctrl_keys_string + ['no_str']
ctrl_keys_long = ctrl_keys_string + [
    'units',
    'display_llim', 'display_hlim',
    'warn_llim', 'warn_hlim',
    'alarm_llim', 'alarm_hlim',
    'ctrl_llim', 'ctrl_hlim'
]
ctrl_keys_double = ctrl_keys_long + ['precision']
ctrl_keys = {
    'PYCA:TEST:LONG': ctrl_keys_long,
    'PYCA:TEST:DOUBLE': ctrl_keys_double,
    'PYCA:TEST:STRING': ctrl_keys_string,
    'PYCA:TEST:ENUM': ctrl_keys_enum,
    'PYCA:TEST:WAVE_LONG': ctrl_keys_long,
    'PYCA:TEST:WAVE_DOUBLE': ctrl_keys_double
}

all_pvs = list(pvdb.keys())
waveform_pvs = list(x for x in pvdb.keys() if x.find('WAVE') >= 0)


# A derived class is strictly needed for pcaspy
class TestDriver(Driver):
    def __init__(self):
        super(TestDriver, self).__init__()


def calc_new_value(name, old_value):
    if isinstance(old_value, int) or isinstance(old_value, long) or isinstance(old_value, float):
        new_value = old_value + 1
    elif isinstance(old_value, str):
        new_value = old_value + 'x'
    elif isinstance(old_value, tuple):
        new_value = tuple([old_value[0] + 1] * len(old_value))
    return new_value


@pytest.fixture(scope='session')
def server():
    server = SimpleServer()
    server.createPV('', pvdb)
    server_thread = ServerThread(server)
    server_thread.start()
    driver = TestDriver()
    yield server
    driver = None
    server_thread.stop()
