# pyca

PyCA (Python Channel Access) is a module that offers lightweight bindings for Python applications
to access EPICS PVs. It acts as a channel access client, much like pyepics. The intention of the
module is to provide better performance for embedded applications, rather than to provide an
interactive interface. The most significant gains will be found when monitoring large waveforms
that need to be processed before exposing them the Python layer.

## Running the test

To run the tests install the _tox_ package. Then run the `tox` command.

If you want to run the tests against an EPICS IOC you have to build the ioc in the
`tests/ioc` directory. You propably need to modify the RELEASE file to point to your EPICS
base installation. After that you can run `tox -- --ioc`.
