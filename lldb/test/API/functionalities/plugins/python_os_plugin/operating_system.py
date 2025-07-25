import lldb
import struct

from lldb.plugins.operating_system import OperatingSystem


class OperatingSystemPlugIn(OperatingSystem):
    """Class that provides data for an instance of a LLDB 'OperatingSystemPython' plug-in class"""

    def __init__(self, process):
        super().__init__(process)

    def create_thread(self, tid, context):
        if tid == 0x444444444:
            thread_info = {
                "tid": tid,
                "name": "four",
                "queue": "queue4",
                "state": "stopped",
                "stop_reason": "none",
            }
            self.threads.append(thread_info)
            return thread_info
        return None

    def get_thread_info(self):
        if self.process.state != lldb.eStateStopped:
            print("Error: get_thread_info called with state not stopped")
            return []

        if not self.threads:
            self.threads = [
                {
                    "tid": 0x111111111,
                    "name": "one",
                    "queue": "queue1",
                    "state": "stopped",
                    "stop_reason": "breakpoint",
                },
                {
                    "tid": 0x222222222,
                    "name": "two",
                    "queue": "queue2",
                    "state": "stopped",
                    "stop_reason": "none",
                },
                {
                    "tid": 0x333333333,
                    "name": "three",
                    "queue": "queue3",
                    "state": "stopped",
                    "stop_reason": "trace",
                },
            ]
        return self.threads

    def get_register_info(self):
        if self.registers is None:
            self.registers = dict()
            self.registers["sets"] = ["GPR"]
            self.registers["registers"] = [
                {
                    "name": "rax",
                    "bitsize": 64,
                    "offset": 0,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "gcc": 0,
                    "dwarf": 0,
                },
                {
                    "name": "rbx",
                    "bitsize": 64,
                    "offset": 8,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "gcc": 3,
                    "dwarf": 3,
                },
                {
                    "name": "rcx",
                    "bitsize": 64,
                    "offset": 16,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "gcc": 2,
                    "dwarf": 2,
                    "generic": "arg4",
                    "alt-name": "arg4",
                },
                {
                    "name": "rdx",
                    "bitsize": 64,
                    "offset": 24,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "gcc": 1,
                    "dwarf": 1,
                    "generic": "arg3",
                    "alt-name": "arg3",
                },
                {
                    "name": "rdi",
                    "bitsize": 64,
                    "offset": 32,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "gcc": 5,
                    "dwarf": 5,
                    "generic": "arg1",
                    "alt-name": "arg1",
                },
                {
                    "name": "rsi",
                    "bitsize": 64,
                    "offset": 40,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "gcc": 4,
                    "dwarf": 4,
                    "generic": "arg2",
                    "alt-name": "arg2",
                },
                {
                    "name": "rbp",
                    "bitsize": 64,
                    "offset": 48,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "gcc": 6,
                    "dwarf": 6,
                    "generic": "fp",
                    "alt-name": "fp",
                },
                {
                    "name": "rsp",
                    "bitsize": 64,
                    "offset": 56,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "gcc": 7,
                    "dwarf": 7,
                    "generic": "sp",
                    "alt-name": "sp",
                },
                {
                    "name": "r8",
                    "bitsize": 64,
                    "offset": 64,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "gcc": 8,
                    "dwarf": 8,
                    "generic": "arg5",
                    "alt-name": "arg5",
                },
                {
                    "name": "r9",
                    "bitsize": 64,
                    "offset": 72,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "gcc": 9,
                    "dwarf": 9,
                    "generic": "arg6",
                    "alt-name": "arg6",
                },
                {
                    "name": "r10",
                    "bitsize": 64,
                    "offset": 80,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "gcc": 10,
                    "dwarf": 10,
                },
                {
                    "name": "r11",
                    "bitsize": 64,
                    "offset": 88,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "gcc": 11,
                    "dwarf": 11,
                },
                {
                    "name": "r12",
                    "bitsize": 64,
                    "offset": 96,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "gcc": 12,
                    "dwarf": 12,
                },
                {
                    "name": "r13",
                    "bitsize": 64,
                    "offset": 104,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "gcc": 13,
                    "dwarf": 13,
                },
                {
                    "name": "r14",
                    "bitsize": 64,
                    "offset": 112,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "gcc": 14,
                    "dwarf": 14,
                },
                {
                    "name": "r15",
                    "bitsize": 64,
                    "offset": 120,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "gcc": 15,
                    "dwarf": 15,
                },
                {
                    "name": "rip",
                    "bitsize": 64,
                    "offset": 128,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "gcc": 16,
                    "dwarf": 16,
                    "generic": "pc",
                    "alt-name": "pc",
                },
                {
                    "name": "rflags",
                    "bitsize": 64,
                    "offset": 136,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                    "generic": "flags",
                    "alt-name": "flags",
                },
                {
                    "name": "cs",
                    "bitsize": 64,
                    "offset": 144,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                },
                {
                    "name": "fs",
                    "bitsize": 64,
                    "offset": 152,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                },
                {
                    "name": "gs",
                    "bitsize": 64,
                    "offset": 160,
                    "encoding": "uint",
                    "format": "hex",
                    "set": 0,
                },
            ]
        return self.registers

    def get_register_data(self, tid):
        return struct.pack(
            "21Q",
            tid + 1,
            tid + 2,
            tid + 3,
            tid + 4,
            tid + 5,
            tid + 6,
            tid + 7,
            tid + 8,
            tid + 9,
            tid + 10,
            tid + 11,
            tid + 12,
            tid + 13,
            tid + 14,
            tid + 15,
            tid + 16,
            tid + 17,
            tid + 18,
            tid + 19,
            tid + 20,
            tid + 21,
        )
