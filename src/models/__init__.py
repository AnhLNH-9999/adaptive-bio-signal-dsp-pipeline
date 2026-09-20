"""Neural-network branch (lazy imports: torch / brevitas are optional dependencies)."""

__all__ = ["EEGNetLite", "QuantEEGNetLite"]


def __getattr__(name):
    if name == "EEGNetLite":
        from .eegnet_lite import EEGNetLite
        return EEGNetLite
    if name == "QuantEEGNetLite":
        from .qeegnet_lite import QuantEEGNetLite
        return QuantEEGNetLite
    raise AttributeError(name)
