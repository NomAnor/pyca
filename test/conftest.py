import sys
import os
import time
import subprocess
from pcaspy import Driver, SimpleServer
from pcaspy.tools import ServerThread
import pytest


IOC_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ioc')

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

def pytest_addoption(parser):
    parser.addoption("--ioc", action="store_true", default=False, help="use real ioc")

def pytest_runtest_setup(item):
    iocmark = item.get_marker("ioc")
    if iocmark is not None:
        if not item.config.getoption('ioc'):
            pytest.skip("test requires epics ioc")


# A derived class is strictly needed for pcaspy
class TestDriver(Driver):
    def __init__(self):
        super(TestDriver, self).__init__()


def calc_new_value(name, old_value):
    if isinstance(old_value, int) or isinstance(old_value, float):
        new_value = old_value + 1
    elif isinstance(old_value, str):
        new_value = old_value + 'x'
    elif isinstance(old_value, list):
        new_value = [old_value[0] + 1] * len(old_value)
    return new_value


@pytest.fixture(scope='session')
def server(pytestconfig):
    if pytestconfig.getoption('ioc'):
        server = subprocess.Popen([os.path.join(IOC_DIR, 'iocBoot/ioctest/st.cmd')], cwd=os.path.join(IOC_DIR, 'iocBoot/ioctest'))
        time.sleep(1)
        yield server
        server.terminate()
    else:
        server = SimpleServer()
        server.createPV('', pvdb)
        server_thread = ServerThread(server)
        server_thread.start()
        driver = TestDriver()
        yield server
        driver = None
        server_thread.stop()

