"""Where Deck Monitor receives the decks: one value (the protocol and its place), the network interfaces to pick from,
and the receiver it makes. Each receiver keeps its decks' newest states on a thread of its own: ``latest()`` gives
(each deck's DeckState, the last sender, packets refused), ``close()`` lets its sockets go."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Optional

OSC, PRODJLINK = "osc", "prodjlink"
PROTOCOLS = (OSC, PRODJLINK)
PROTOCOL_NAMES = {OSC: "OSC", PRODJLINK: "Pro DJ Link"}     # product names: not translated


@dataclass(frozen=True)
class Where:
    """OSC (Basic and Full) at ``host:port``, or Pro DJ Link on the network of ``interface`` (a local IPv4
    address; "": the first interface that has one, not the loopback)."""
    protocol: str = OSC
    host: str = "127.0.0.1"
    port: int = 39321
    interface: str = ""


@dataclass(frozen=True)
class Interface:
    name: str
    address: str          # IPv4
    broadcast: str        # its network's broadcast address
    mac: bytes = b""      # 6 bytes ("": not known)


def interfaces() -> list[Interface]:
    """The computer's network interfaces with an IPv4 address (the loopback and link-local last), as the operating
    system names them."""
    try:
        import psutil
        addresses = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
    except Exception:
        return []
    found = []
    for name, addrs in addresses.items():
        if name in stats and not stats[name].isup:
            continue
        mac = next((a.address for a in addrs if a.family == getattr(psutil, "AF_LINK", -1)), "")
        mac_bytes = bytes.fromhex(mac.replace("-", "").replace(":", "")) if mac else b""
        for a in addrs:
            if a.family != socket.AF_INET or not a.address:
                continue
            try:
                net = ipaddress.IPv4Network(f"{a.address}/{a.netmask or '255.255.255.255'}", strict=False)
            except ValueError:
                continue
            found.append(Interface(name, a.address, str(net.broadcast_address),
                                   mac_bytes if len(mac_bytes) == 6 else b""))
    rank = lambda i: (ipaddress.IPv4Address(i.address).is_loopback, ipaddress.IPv4Address(i.address).is_link_local)
    return sorted(found, key=rank)


def interface_for(address: str) -> Optional[Interface]:
    """The interface ``address`` names ("": the first, the loopback and link-local last); None: none has it."""
    found = interfaces()
    if not address:
        return found[0] if found else None
    return next((i for i in found if i.address == address), None)


def open_receiver(where: Where):
    """The receiver ``where`` names (raises OSError, ValueError when it cannot listen there)."""
    if where.protocol == PRODJLINK:
        from core.deck_monitor.prodjlink import ProDjLinkReceiver
        return ProDjLinkReceiver(where.interface)
    from core.deck_monitor.osc_basic import OscBasicReceiver
    return OscBasicReceiver(str(where.host or "").strip() or "127.0.0.1", int(where.port))
