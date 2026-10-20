"""The grading host's Petri drive's tunnel is judged by the EXACT node and proxy port it forwards to, re-created when stale, and torn down on
exit (2026-10-03). petri_drive_full22.sh used to accept any holder of its local port whose command line contained `" $NODE"`, so
with NODE=gpu-2 the previous attempt's tunnel to gpu-20 was "reused" and /v1/models was read off a dead forward; and the tunnel a
drive opened outlived it. The verdict now lives in scripts/sdfbase/full22/petri_tunnel.sh, sourced by the drive:

* `f22_tunnel_holder_verdict` — reuse only for an ssh carrying ONE forward that binds the drive's local port to exactly
  (127.0.0.1, proxy port) on exactly the node; stale for one forward elsewhere (a node whose name merely has the drive's as a prefix, the
  same node at the re-run pass's port, the `-J` hop mistaken for the node, a non-loopback destination); ambiguous for a line carrying more
  than one forward (the verdict used to keep the LAST -L and ignore the local port, so a two-forward ssh whose forward on OUR port went to
  the re-run pass's proxy read `reuse`) — the drive refuses it by name, neither reusing nor closing it; foreign otherwise, a forward
  that does not bind our local port included;
* `f22_tunnel_close_stale` — kills the forward by the pid its line starts with and waits for the local port (a real listener here);
* `f22_tunnel_teardown` — kills the tunnel THIS run opened, once, and is a no-op when the tunnel was reused;
* `f22_tunnel_holder_line` — the holder of the local port is the LISTENING SOCKET's owner (ss -ltnp) with its argv (before 2026-10-03 the
  drive's `pgrep -fa "ssh .*-L <port>:127.0.0.1:"` never found the bind-address or glued forms the verdict accepts, so a legitimate tunnel in
  either form left the port "held by something else" and the unattended drive down); every accepted form is discovered and reused, a stale
  one discovered and closed, a non-ssh owner refused — against REAL listeners whose argv is set with `exec -a ssh`;
* the drive sources the helper, traps the teardown on EXIT, uses the verdict, discovers the holder from the socket (no tunnel pgrep anywhere), and
  no longer carries the substring glob or an inline kill.

Synthetic command lines only; no ssh runs. TESTHOST-free: node names are the clusters' generic `node-N` spelling."""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from _env import have_exe, ipv6_loopback_available

REPO = Path(__file__).resolve().parents[1]
F22 = REPO / "scripts" / "sdfbase" / "full22"
HELPER = F22 / "petri_tunnel.sh"
DRIVER_PATH = F22 / "petri_drive_full22.sh"
DRIVER = DRIVER_PATH.read_text()

SSH = ("ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new "
       "-o BatchMode=yes")


def _bash(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script, "_", str(HELPER), *args], capture_output=True, text=True)


def _verdict(line: str, node: str, pport: int, lport: int = 18972) -> str:
    r = _bash('source "$1"; f22_tunnel_holder_verdict "$2" "$3" "$4" "$5"', line, node, str(pport), str(lport))
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


@pytest.mark.parametrize("line,node,pport,want", [
    (f"4242 {SSH} -L 18972:127.0.0.1:8972 -J login-0 gpu-20", "gpu-2", 8972, "stale"),    # the drive's node is a prefix of the tunnel's target
    (f"4242 {SSH} -L 18972:127.0.0.1:8972 -J login-0 gpu-2", "gpu-2", 8972, "reuse"),
    (f"4242 {SSH} -L 18972:127.0.0.1:8972 -J login-0 gpu-2", "gpu-20", 8972, "stale"),
    (f"4242 {SSH} -L 18972:127.0.0.1:9072 -J login-0 gpu-2", "gpu-2", 8972, "stale"),     # same node, the re-run pass's port
    (f"4242 {SSH} -L 18972:127.0.0.1:8972 login-0", "login-0", 8972, "reuse"),              # proxy on the jump host: no -J
    (f"4242 {SSH} -L 18972:127.0.0.1:8972 -J login-0 gpu-2", "login-0", 8972, "stale"),   # the jump host is not the node
    (f"4242 {SSH} -L18972:127.0.0.1:8972 -J login-0 gpu-2", "gpu-2", 8972, "reuse"),      # glued -L<spec>
    (f"4242 /usr/bin/ssh -L 18972:127.0.0.1:8972 gpu-2", "gpu-2", 8972, "reuse"),         # a full ssh path
    (f"4242 {SSH} -L 127.0.0.1:18972:127.0.0.1:8972 -J login-0 gpu-2", "gpu-2", 8972, "reuse"),  # the bind-address form of the same forward
    (f"4242 {SSH} -L 18972:gpu-3:8972 -J login-0 gpu-2", "gpu-2", 8972, "stale"),         # right node and port, but not the node's loopback
    # more than one forward — never the drive's tunnel, whichever forward matches
    (f"4242 {SSH} -L 18972:127.0.0.1:9072 -L 18973:127.0.0.1:8972 -J login-0 gpu-2", "gpu-2", 8972, "ambiguous"),  # OUR port -> the re-run proxy; the LAST -L -> this one (the old verdict: reuse)
    (f"4242 {SSH} -L 18973:127.0.0.1:8972 -L 18972:127.0.0.1:8972 -J login-0 gpu-2", "gpu-2", 8972, "ambiguous"),  # even with a right forward on our port
    (f"4242 {SSH} -L18972:127.0.0.1:9072 -L18973:127.0.0.1:8972 gpu-2", "gpu-2", 8972, "ambiguous"),              # glued, same thing
    (f"4242 {SSH} -L 18973:127.0.0.1:8972 -J login-0 gpu-2", "gpu-2", 8972, "foreign"),    # one forward, not on our port (the socket's owner forwards another)
    (f"4242 {SSH} -L 18972:127.0.0.1 -J login-0 gpu-2", "gpu-2", 8972, "foreign"),         # a malformed forward
    ("4242 python3 -m http.server 18972", "gpu-2", 8972, "foreign"),
    ("4242 ssh gpu-2", "gpu-2", 8972, "foreign"),                                          # ssh without a forward
    ("", "gpu-2", 8972, "foreign"),
])
def test_the_holder_verdict_is_by_exact_node_and_proxy_port(line, node, pport, want):
    assert _verdict(line, node, pport) == want


@pytest.mark.parametrize("lport,want", [(18972, "reuse"), (18973, "foreign"), (8972, "foreign")])
def test_the_verdict_is_for_the_local_port_the_drive_asked_about(lport, want):
    """The same exact forward is `reuse` only for the local port it binds; asked about another port (the proxy port's own number
    included, which the old verdict never compared) it is foreign — the line does not hold the port the drive is about to use."""
    assert _verdict(f"4242 {SSH} -L 18972:127.0.0.1:8972 -J login-0 gpu-2", "gpu-2", 8972, lport) == want


def test_close_stale_refuses_a_line_that_starts_with_no_pid():
    r = _bash('source "$1"; f22_tunnel_close_stale "$2" 18972', "not-a-pid ssh -L 18972:127.0.0.1:8972 gpu-20")
    assert r.returncode == 1
    r = _bash('source "$1"; f22_tunnel_close_stale "" 18972')
    assert r.returncode == 1


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_close_stale_kills_the_forward_by_its_pid_and_waits_for_the_port():
    """A real listener stands in for the stale ssh (same pid-first line shape); the helper kills it by pid and returns 0 once the port is free."""
    port = _free_port()
    srv = subprocess.Popen([sys.executable, "-c",
                            "import socket,time; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
                            f"s.bind(('127.0.0.1', {port})); s.listen(1); time.sleep(120)"])
    try:
        for _ in range(100):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.05)
        line = f"{srv.pid} {SSH} -L {port}:127.0.0.1:8972 -J login-0 gpu-20"
        r = _bash('source "$1"; f22_tunnel_close_stale "$2" "$3"', line, str(port))
        assert r.returncode == 0, (r.stdout, r.stderr)
        assert srv.wait(timeout=10) is not None
        with socket.socket() as probe:
            assert probe.connect_ex(("127.0.0.1", port)) != 0
    finally:
        if srv.poll() is None:
            os.kill(srv.pid, signal.SIGKILL)


def test_teardown_kills_only_the_tunnel_this_run_opened_and_is_idempotent():
    r = _bash('source "$1"; log(){ echo "LOG $*"; }; sleep 300 & TUNNEL_PID=$!; p=$TUNNEL_PID; '
              'f22_tunnel_teardown; kill -0 "$p" 2>/dev/null && echo alive || echo dead; echo "pid=[$TUNNEL_PID]"; '
              'f22_tunnel_teardown; echo "rc=$?"')
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "dead" in out and "pid=[]" in out and "rc=0" in out
    assert out.count("LOG tunnel (pid") == 1, out                      # the second call logs nothing: a reused / already-closed tunnel
    r = _bash('source "$1"; TUNNEL_PID=""; f22_tunnel_teardown; echo "rc=$?"')
    assert r.stdout.strip() == "rc=0"                                   # a reused tunnel (none opened here) is left to its opener


def test_the_drive_sources_the_helper_traps_the_teardown_and_carries_no_substring_match():
    assert '. "$(dirname "${BASH_SOURCE[0]}")/petri_tunnel.sh"' in DRIVER
    assert "trap 'f22_tunnel_teardown' EXIT" in DRIVER
    assert 'f22_tunnel_holder_verdict "$OLD" "$NODE" "$PPORT" "$LOCALPORT"' in DRIVER   # The verdict is for THIS local port
    assert 'OLD=$(f22_tunnel_holder_line "$LOCALPORT")' in DRIVER            # The holder is the SOCKET's owner, not a pgrep over one -L spelling
    assert "pgrep -fa" not in DRIVER and DRIVER.count("pgrep") == 1        # … anywhere (the one left is petri_run's own-pid comment); the stale-close FATAL names the socket's holder too
    assert DRIVER.count("ambiguous)") == 1                               # the multi-forward refusal arm
    assert 'f22_tunnel_close_stale "$OLD" "$LOCALPORT"' in DRIVER
    assert '*" $NODE"*)' not in DRIVER                                  # the glob that matched gpu-20 for gpu-2
    assert 'kill "$TUNNEL_PID"' not in DRIVER                           # one teardown path, in the helper
    assert DRIVER.count("f22_tunnel_teardown") >= 3                     # the trap + both release paths
    assert "proxy and tunnel left up" not in DRIVER                     # the legs-remain path no longer leaves the tunnel up
    for f in (DRIVER_PATH, HELPER):
        assert subprocess.run(["bash", "-n", str(f)], capture_output=True).returncode == 0, f


MULTI_WRONG = f"4242 {SSH} -L 18972:127.0.0.1:9072 -L 18973:127.0.0.1:8972 -J login-0 gpu-2"   # OUR port -> the re-run pass's proxy; the other forward -> this proxy


def _drive_tunnel_block() -> str:
    """The drive's holder decision VERBATIM: from `OPEN_TUNNEL=1` up to the line that would open a tunnel (ss / the holder discovery / log
    stubbed here; `test_the_drives_holder_discovery_reuses_a_live_bind_address_tunnel` runs it against a real socket with nothing stubbed but
    log and the close)."""
    start = DRIVER.index("OPEN_TUNNEL=1\n")
    end = DRIVER.index('if [ "$OPEN_TUNNEL" = 1 ]; then', start)
    return DRIVER[start:end]


def _run_drive_block(holder_line: str) -> subprocess.CompletedProcess:
    script = ('source "$1"; LOCALPORT=18972; NODE=gpu-2; PPORT=8972; log(){ echo "LOG $*"; }; '
              'ss(){ echo "LISTEN 0 4096 127.0.0.1:18972 0.0.0.0:*"; }; f22_tunnel_holder_line(){ printf "%s\\n" "$HOLDER_LINE"; }; '
              'f22_tunnel_close_stale(){ echo "CLOSE-STALE $*"; }\n'               # never kill anything from a test: the close is stubbed
              + _drive_tunnel_block() + '\necho "OPEN_TUNNEL=$OPEN_TUNNEL"')
    return subprocess.run(["bash", "-c", script, "_", str(HELPER)], capture_output=True, text=True,
                          env={**os.environ, "HOLDER_LINE": holder_line})


def test_the_drive_refuses_a_multi_forward_holder_whose_forward_on_our_port_goes_elsewhere_by_name():
    """Integrated, on the drive's own case statement: a holder of our local port carrying two forwards — ours to the re-run pass's
    proxy port, another local port to THIS proxy on the same node. The old verdict kept the last -L and said `reuse`; the drive would have
    read /v1/models through a forward to the wrong proxy. Now: FATAL by name, exit 1, nothing reused, nothing closed."""
    r = _run_drive_block(MULTI_WRONG)
    assert r.returncode == 1, (r.stdout, r.stderr)
    assert "FATAL" in r.stdout and "more than one forward" in r.stdout and MULTI_WRONG in r.stdout, r.stdout
    assert "reusing" not in r.stdout and "CLOSE-STALE" not in r.stdout and "OPEN_TUNNEL=" not in r.stdout, r.stdout


def test_the_drive_reuses_an_exact_single_forward_and_recreates_a_stale_one():
    r = _run_drive_block(f"4242 {SSH} -L 18972:127.0.0.1:8972 -J login-0 gpu-2")
    assert r.returncode == 0 and "reusing" in r.stdout and "OPEN_TUNNEL=0" in r.stdout, (r.stdout, r.stderr)
    r = _run_drive_block(f"4242 {SSH} -L 18972:127.0.0.1:8972 -J login-0 gpu-20")                  # the stale case: close, re-create
    assert r.returncode == 0 and "CLOSE-STALE 4242" in r.stdout and "OPEN_TUNNEL=1" in r.stdout, (r.stdout, r.stderr)
    r = _run_drive_block("4242 python3 -m http.server 18972")                                       # foreign: refused, not closed
    assert r.returncode == 1 and "held by something else" in r.stdout and "CLOSE-STALE" not in r.stdout, r.stdout


LISTEN = ("import socket,time; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
          "s.bind(('127.0.0.1', {port})); s.listen(1); time.sleep(120)")        # carries no '-L': the verdict tokenises the whole argv


def _listener(port: int, *argv: str, as_ssh: bool = True) -> subprocess.Popen:
    """A REAL listener on 127.0.0.1:port whose argv reads `ssh -c <code> <argv…>` (exec -a: the holder discovery reads the socket owner's argv,
    and this is what a forward looks like to it) — or a plain `python3 -c <code>` when as_ssh is False (a foreign owner)."""
    code = LISTEN.format(port=port)
    if as_ssh:
        srv = subprocess.Popen(["bash", "-c", 'exec -a ssh "$0" -c "$1" "${@:2}"', sys.executable, code, *argv])
    else:
        srv = subprocess.Popen([sys.executable, "-c", code])
    for _ in range(200):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return srv
        time.sleep(0.05)
    srv.kill()
    raise AssertionError("the listener never came up")


def _stop(srv: subprocess.Popen) -> None:
    if srv.poll() is None:
        srv.kill()
    srv.wait(timeout=10)


SSH_OPTS = ["-N", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6", "-o", "ExitOnForwardFailure=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]

# The tests below discover a REAL listener's owner through the helper, which runs `ss -ltnp` (iproute2) with its stderr discarded: on a
# machine without ss the holder line is simply empty (no error), the verdict foreign, and the drive's block FATALs "held by something else",
# so each one skips on the binary's absence, per test, never the module (the verdict and parsing tests stub ss and run everywhere). The two
# that bind a second listener on [::1] also need an IPv6 loopback, which a minimal container's network namespace may not carry.
NEEDS_SS = pytest.mark.skipif(not have_exe("ss"), reason="the holder discovery runs ss -ltnp (iproute2)")
NEEDS_IPV6 = pytest.mark.skipif(not ipv6_loopback_available(), reason="binds a second listener on the IPv6 loopback [::1]")


@NEEDS_SS
@pytest.mark.parametrize("form,want", [("plain", "reuse"), ("bind", "reuse"), ("glued", "reuse"), ("stale", "stale"), ("foreign", "foreign"), ("long", "reuse")])
def test_the_holder_is_discovered_from_the_listening_socket_in_every_accepted_forward_form(form, want):
    """Discovery + verdict end to end against a live socket. The old pgrep found only the plain form; the bind-address and glued forms
    read as "held by something else" with no holder line at all. The `long` form is the drive's own tunnel shape — every -o option in front
    of the forward, an argv well past 80 columns: chain 294 attempt 1 read it through `ps -o args=`, which truncates at the terminal width
    under a captured stdout and lost the -L token; the holder's argv is read from /proc/<pid>/cmdline, exact and unbounded."""
    port = _free_port()
    argv = {"plain": ["-L", f"{port}:127.0.0.1:8972", "-J", "login-0", "gpu-2"],
            "bind": ["-L", f"127.0.0.1:{port}:127.0.0.1:8972", "-J", "login-0", "gpu-2"],
            "glued": [f"-L{port}:127.0.0.1:8972", "-J", "login-0", "gpu-2"],
            "stale": ["-L", f"{port}:127.0.0.1:8972", "-J", "login-0", "gpu-20"],
            "foreign": [],
            "long": [*SSH_OPTS, *[x for i in range(12) for x in ("-o", f"SetEnv=PCR_PAD_{i}=xxxxxxxxxxxxxxxx")], "-L", f"{port}:127.0.0.1:8972", "-J", "login-0", "gpu-2"]}[form]
    srv = _listener(port, *argv, as_ssh=(form != "foreign"))
    try:
        r = _bash('source "$1"; line=$(f22_tunnel_holder_line "$2"); echo "LINE=$line"; echo "VERDICT=$(f22_tunnel_holder_verdict "$line" gpu-2 8972 "$2")"', str(port))
        assert r.returncode == 0, r.stderr
        line = r.stdout.split("LINE=", 1)[1].split("\n", 1)[0]
        assert line.startswith(f"{srv.pid} "), (line, srv.pid)                                  # the socket owner's pid first: the verdict's line shape
        for a in argv:
            assert a in line.split(), (a, line)
        if form == "long":
            assert len(line) > 400 and line.rstrip().endswith(" gpu-2"), len(line)             # unbounded: the whole argv, the node last
        assert f"VERDICT={want}" in r.stdout, r.stdout
    finally:
        _stop(srv)
    r = _bash('source "$1"; f22_tunnel_holder_line "$2"; echo "rc=$?"', str(port))             # the port is free again: nothing, rc 0
    assert r.stdout.strip() == "rc=0", r.stdout


@NEEDS_SS
def test_the_drives_holder_discovery_reuses_a_live_bind_address_tunnel():
    """Discovery on the drive's OWN block, nothing stubbed but log and the close: a real listener whose argv is the bind-address form is discovered
    (ss) and reused. Before the fix OLD was empty for this form and the drive FATALed "held by something else"."""
    port = _free_port()
    srv = _listener(port, "-L", f"127.0.0.1:{port}:127.0.0.1:8972", "-J", "login-0", "gpu-2")
    try:
        script = (f'source "$1"; LOCALPORT={port}; NODE=gpu-2; PPORT=8972; log(){{ echo "LOG $*"; }}; f22_tunnel_close_stale(){{ echo "CLOSE-STALE $*"; }}\n'
                  + _drive_tunnel_block() + '\necho "OPEN_TUNNEL=$OPEN_TUNNEL"')
        r = subprocess.run(["bash", "-c", script, "_", str(HELPER)], capture_output=True, text=True)
        assert r.returncode == 0 and f"reusing" in r.stdout and "OPEN_TUNNEL=0" in r.stdout, (r.stdout, r.stderr)
        assert "CLOSE-STALE" not in r.stdout
    finally:
        _stop(srv)


LISTEN6 = ("import socket,time; s=socket.socket(socket.AF_INET6); s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1); "
           "s.bind(('::1', {port})); s.listen(1); time.sleep(120)")


def _listener6(port: int) -> subprocess.Popen:
    """A SECOND process listening on [::1]:port (IPv6 loopback only): a distinct owner at the same number."""
    srv = subprocess.Popen([sys.executable, "-c", LISTEN6.format(port=port)])
    for _ in range(200):
        with socket.socket(socket.AF_INET6) as probe:
            if probe.connect_ex(("::1", port)) == 0:
                return srv
        time.sleep(0.05)
    srv.kill()
    raise AssertionError("the IPv6 listener never came up")


def _holder_line(port: int) -> str:
    r = _bash('source "$1"; f22_tunnel_holder_line "$2"; echo "rc=$?"', str(port))
    assert r.returncode == 0, r.stderr
    return r.stdout


@NEEDS_SS
@NEEDS_IPV6
def test_two_distinct_owners_at_one_port_number_are_no_holder_and_the_drive_refuses_without_killing():
    """tunnel-fix-4: an ssh-shaped listener on 127.0.0.1:PORT and ANOTHER process on [::1]:PORT. The first-pid read could pick either; now the
    sockets at the number belong to two pids, so the holder line is empty, the verdict foreign, and the drive's block FATALs with nothing closed."""
    port = _free_port()
    a = _listener(port, "-L", f"{port}:127.0.0.1:8972", "-J", "login-0", "gpu-20")        # a STALE-looking forward: the old read could have killed it
    b = _listener6(port)
    try:
        assert _holder_line(port).strip() == "rc=0"                                           # nothing: two distinct owners
        script = (f'source "$1"; LOCALPORT={port}; NODE=gpu-2; PPORT=8972; log(){{ echo "LOG $*"; }}; f22_tunnel_close_stale(){{ echo "CLOSE-STALE $*"; }}\n'
                  + _drive_tunnel_block() + '\necho "OPEN_TUNNEL=$OPEN_TUNNEL"')
        r = subprocess.run(["bash", "-c", script, "_", str(HELPER)], capture_output=True, text=True)
        assert r.returncode == 1 and "held by something else" in r.stdout and "CLOSE-STALE" not in r.stdout, (r.stdout, r.stderr)
        assert a.poll() is None and b.poll() is None                                          # both still alive: nothing was killed
    finally:
        _stop(a); _stop(b)


@NEEDS_SS
@NEEDS_IPV6
def test_one_ssh_bound_on_both_loopbacks_is_still_one_holder_and_an_ipv6_only_listener_is_none():
    """The common dual-stack bind of ONE process (127.0.0.1 and ::1, one pid) stays one holder, read from the IPv4 loopback socket; a listener
    on ::1 alone has no IPv4 loopback socket to read and is no holder (the drive binds 127.0.0.1)."""
    port = _free_port()
    both = ("import socket,time; a=socket.socket(); a.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); a.bind(('127.0.0.1', {port})); a.listen(1); "
            "b=socket.socket(socket.AF_INET6); b.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1); b.bind(('::1', {port})); b.listen(1); time.sleep(120)")
    srv = subprocess.Popen(["bash", "-c", 'exec -a ssh "$0" -c "$1" "${@:2}"', sys.executable, both.format(port=port),
                            "-L", f"{port}:127.0.0.1:8972", "-J", "login-0", "gpu-2"])
    try:
        for _ in range(200):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.05)
        line = _holder_line(port).split("\n", 1)[0]
        assert line.startswith(f"{srv.pid} ") and f"{port}:127.0.0.1:8972" in line.split(), line
        r = _bash('source "$1"; f22_tunnel_holder_verdict "$(f22_tunnel_holder_line "$2")" gpu-2 8972 "$2"', str(port))
        assert r.stdout.strip() == "reuse", r.stdout
    finally:
        _stop(srv)
    port6 = _free_port()
    only6 = _listener6(port6)
    try:
        assert _holder_line(port6).strip() == "rc=0"                                          # no IPv4 loopback socket: no holder
    finally:
        _stop(only6)


def test_an_owner_ss_cannot_show_is_no_holder():
    """Another user's listener shows in ss without a users:(...) part (no pid visible): the holder line is empty and the drive refuses it as
    foreign — never a guess at a pid. ss is stubbed to that shape (a real other-user socket cannot be made from a test)."""
    r = _bash('source "$1"; ss(){ echo "LISTEN 0 128 127.0.0.1:18960 0.0.0.0:*"; }; f22_tunnel_holder_line 18960; echo "rc=$?"')
    assert r.stdout.strip() == "rc=0", r.stdout
    r = _bash('source "$1"; ss(){ echo "LISTEN 0 128 127.0.0.1:18960 0.0.0.0:* users:(("ssh",pid=4242,fd=4))"; echo "LISTEN 0 128 [::1]:18960 [::]:* users:(("sshd",pid=4343,fd=5))"; }; '
              'f22_tunnel_holder_line 18960; echo "rc=$?"')
    assert r.stdout.strip() == "rc=0", r.stdout                                               # two visible owners: nothing, however the pids read


def test_the_helper_header_names_the_exact_argv_source_and_the_one_owner_rule():
    src = HELPER.read_text()
    head = src[:src.index("f22_tunnel_holder_line() {")]
    assert "/proc/<pid>/cmdline" in head and "ps -ww -o args=" in head and "ps -o args=)" not in head
    assert "MORE THAN ONE pid" in head and "127.0.0.1:LOCALPORT" in head
