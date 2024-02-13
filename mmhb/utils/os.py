import platform


def detect_os():
    """Detects the operating system.

    Returns:
        str: 'Windows', 'Mac', or 'Linux' based on the operating system.
    """
    os_name = platform.system()
    assert os_name in ["Darwin", "Windows", "Linux"], "OS not supported"
    if os_name == "Darwin":
        return "mac"
    elif os_name == "Windows":
        return "windows"
    elif os_name == "Linux":
        return "linux"
