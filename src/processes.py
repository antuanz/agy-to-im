"""Cross-platform subprocess-group creation, termination, and reaping."""
from __future__ import annotations

import asyncio
import os
import signal

PROCESS_STOP_GRACE_S = 2.0


def process_group_options() -> dict[str, object]:
    if os.name == "nt":
        import subprocess

        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def terminate_process_group(proc: asyncio.subprocess.Process) -> None:
    """Terminate the subprocess tree and always reap the direct child."""
    if proc.returncode is not None:
        await proc.wait()
        return
    if os.name == "nt":
        _terminate_windows_descendants(proc.pid)
        proc.kill()
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    exited = False
    try:
        await asyncio.wait_for(proc.wait(), timeout=PROCESS_STOP_GRACE_S)
        exited = True
    except asyncio.TimeoutError:
        pass
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif not exited:
        proc.kill()
    if not exited:
        await proc.wait()


def _terminate_windows_descendants(root_pid: int) -> None:
    """Terminate descendants using the Tool Help API without shell commands."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        return
    parents: dict[int, int] = {}
    entry = ProcessEntry()
    entry.dwSize = ctypes.sizeof(entry)
    try:
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)

    descendants: list[int] = []
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        children = [pid for pid, ppid in parents.items() if ppid == parent]
        descendants.extend(children)
        frontier.extend(children)
    for pid in reversed(descendants):
        handle = kernel32.OpenProcess(0x0001, False, pid)  # PROCESS_TERMINATE
        if handle:
            try:
                kernel32.TerminateProcess(handle, 1)
            finally:
                kernel32.CloseHandle(handle)
