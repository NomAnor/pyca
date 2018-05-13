#!../../bin/linux-x86_64/test

< envPaths

cd "${TOP}"

## Register all support components
dbLoadDatabase("dbd/test.dbd")
test_registerRecordDeviceDriver(pdbbase)

cd "${TOP}/iocBoot/${IOC}"

## Load record instances
dbLoadRecords("records.db")

iocInit
