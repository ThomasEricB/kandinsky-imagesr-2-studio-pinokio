"""Show the operating system's own file dialog and print the picked paths, one per line.

    python dialog.py folder|files "<title>" "<start folder>"

Exit codes: 0 = picked or cancelled (cancel prints nothing), 3 = no native dialog available.
Diagnostics go to stderr. serve.py runs this in a subprocess so a dialog problem can never
take the web server down, and so Windows COM gets a fresh main thread.

Which dialog:
  Linux    the XDG desktop portal, which shows the running desktop's own dialog (KDE on
           Plasma, GNOME on Ubuntu/GNOME, GTK on Xfce/Cinnamon/MATE...). Then kdialog or
           zenity, preferring the one that matches the desktop.
  Windows  the modern Explorer dialog (IFileOpenDialog), also for folders.
  macOS    Finder's dialog through AppleScript.
There is deliberately no Tk fallback: on Linux its dialogs are Tk's own dated windows.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
# Linux dialogs match patterns case-sensitively, so list upper case too.
PATTERNS = [f"*{s}" for s in IMAGE_SUFFIXES] + [f"*{s.upper()}" for s in IMAGE_SUFFIXES]
TIMEOUT = 1800  # seconds a dialog may stay open


class Unavailable(Exception):
    """This dialog can't be shown here; try the next one."""


def note(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- Linux


def portal(folder: bool, title: str, start: str) -> list[str]:
    """org.freedesktop.portal.FileChooser.OpenFile, answered by the desktop's own backend."""
    try:
        from jeepney import DBusAddress, MatchRule, new_method_call
        from jeepney.bus_messages import message_bus
        from jeepney.io.blocking import open_dbus_connection
    except ImportError as error:
        raise Unavailable(f"jeepney missing ({error})") from error
    try:
        connection = open_dbus_connection(bus="SESSION")
    except Exception as error:  # noqa: BLE001 - no session bus
        raise Unavailable(f"no D-Bus session ({error})") from error

    with connection:
        token = "kisr" + secrets.token_hex(6)
        sender = connection.unique_name.lstrip(":").replace(".", "_")
        handle = f"/org/freedesktop/portal/desktop/request/{sender}/{token}"
        rule = MatchRule(type="signal", interface="org.freedesktop.portal.Request",
                         member="Response", path=handle)
        connection.send_and_get_reply(message_bus.AddMatch(rule))
        options = {
            "handle_token": ("s", token),
            "modal": ("b", True),
            "multiple": ("b", not folder),
            "directory": ("b", folder),
            "current_folder": ("ay", os.fsencode(start) + b"\0"),
        }
        if not folder:
            options["filters"] = ("a(sa(us))", [("Images", [(0, p) for p in PATTERNS])])
        address = DBusAddress("/org/freedesktop/portal/desktop", bus_name="org.freedesktop.portal.Desktop",
                              interface="org.freedesktop.portal.FileChooser")
        with connection.filter(rule) as queue:
            reply = connection.send_and_get_reply(new_method_call(address, "OpenFile", "ssa{sv}",
                                                                  ("", title, options)))
            if reply.header.message_type.name == "error":
                raise Unavailable(f"portal refused: {reply.body}")
            response = connection.recv_until_filtered(queue, timeout=TIMEOUT)
    code, results = response.body
    if code == 1:  # cancelled
        return []
    if code != 0:
        raise Unavailable(f"portal answered {code}")
    uris = results.get("uris", ("as", []))[1]
    return [unquote(urlparse(uri).path) for uri in uris if uri.startswith("file://")]


def kdialog(folder: bool, title: str, start: str) -> list[str]:
    command = ["kdialog", "--title", title]
    command += (["--getexistingdirectory", start] if folder else
                ["--getopenfilename", start, f"Images ({' '.join(PATTERNS)})", "--multiple", "--separate-output"])
    return _run_tool(command)


def zenity(folder: bool, title: str, start: str) -> list[str]:
    command = ["zenity", "--file-selection", f"--title={title}", f"--filename={start.rstrip('/')}/"]
    command += ["--directory"] if folder else ["--multiple", "--separator=\n",
                                                f"--file-filter=Images | {' '.join(PATTERNS)}"]
    return _run_tool(command)


def _run_tool(command: list[str]) -> list[str]:
    executable = shutil.which(command[0])
    if not executable:
        raise Unavailable(f"{command[0]} not installed")
    result = subprocess.run([executable, *command[1:]], capture_output=True, text=True, timeout=TIMEOUT)
    if result.returncode not in (0, 1):  # 1 = cancelled
        raise Unavailable(f"{command[0]} exited with {result.returncode}: {result.stderr.strip()[:200]}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def linux(folder: bool, title: str, start: str) -> list[str]:
    if not (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")):
        raise Unavailable("no graphical session")
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
    tools = [kdialog, zenity] if "kde" in desktop else [zenity, kdialog]
    for dialog in (portal, *tools):
        try:
            return dialog(folder, title, start)
        except (Unavailable, OSError, subprocess.SubprocessError) as error:
            note(f"{dialog.__name__}: {error}")
    raise Unavailable("no desktop file dialog found (install xdg-desktop-portal, zenity or kdialog)")


# --------------------------------------------------------------------------- Windows


def windows(folder: bool, title: str, start: str) -> list[str]:
    """The Vista-and-later Explorer dialog (IFileOpenDialog), for files and folders alike."""
    try:
        import pythoncom
        import pywintypes
        from win32com.shell import shell, shellcon
    except ImportError as error:
        raise Unavailable(f"pywin32 missing ({error})") from error

    pythoncom.CoInitialize()
    try:
        dialog = pythoncom.CoCreateInstance(shell.CLSID_FileOpenDialog, None, pythoncom.CLSCTX_INPROC_SERVER,
                                            shell.IID_IFileOpenDialog)
        options = dialog.GetOptions() | shellcon.FOS_FORCEFILESYSTEM
        options |= shellcon.FOS_PICKFOLDERS if folder else shellcon.FOS_ALLOWMULTISELECT | shellcon.FOS_FILEMUSTEXIST
        dialog.SetOptions(options)
        dialog.SetTitle(title)
        if not folder:
            dialog.SetFileTypes([("Images", ";".join(f"*{s}" for s in IMAGE_SUFFIXES))])
        try:
            dialog.SetFolder(shell.SHCreateItemFromParsingName(start, None, shell.IID_IShellItem))
        except pywintypes.com_error:
            pass  # start folder unusable: Windows picks its own
        try:
            dialog.Show(None)
        except pywintypes.com_error as error:
            if error.hresult & 0xFFFFFFFF == 0x800704C7:  # ERROR_CANCELLED
                return []
            raise
        items = dialog.GetResults()
        return [items.GetItemAt(i).GetDisplayName(shellcon.SIGDN_FILESYSPATH) for i in range(items.GetCount())]
    finally:
        pythoncom.CoUninitialize()


# --------------------------------------------------------------------------- macOS


def macos(folder: bool, title: str, start: str) -> list[str]:
    """Finder's open panel through AppleScript (no extra packages)."""
    def quoted(text: str) -> str:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'

    where = f"default location (POSIX file {quoted(start)})"
    if folder:
        script = f"POSIX path of (choose folder with prompt {quoted(title)} {where})"
    else:
        script = (f"set picked to choose file with prompt {quoted(title)} of type {{\"public.image\"}} "
                  f"{where} with multiple selections allowed\n"
                  "set out to \"\"\nrepeat with f in picked\nset out to out & POSIX path of f & linefeed\n"
                  "end repeat\nreturn out")
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=TIMEOUT)
    if result.returncode != 0:
        if "-128" in result.stderr:  # user cancelled
            return []
        raise Unavailable(f"osascript: {result.stderr.strip()[:200]}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


# --------------------------------------------------------------------------- main


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("folder", "files"):
        note(__doc__)
        return 2
    folder = sys.argv[1] == "folder"
    title = sys.argv[2] if len(sys.argv) > 2 else ("Choose a folder of images" if folder else "Choose images")
    start = sys.argv[3] if len(sys.argv) > 3 and Path(sys.argv[3]).is_dir() else str(Path.home())
    show = windows if sys.platform == "win32" else macos if sys.platform == "darwin" else linux
    try:
        paths = show(folder, title, start)
    except Unavailable as error:
        note(str(error))
        return 3
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
