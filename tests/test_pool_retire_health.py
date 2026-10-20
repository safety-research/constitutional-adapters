"""31d-fix-3: the last-server decision counted a surviving `.json` member healthy merely because its batch job was live on
sacct — attacker_pool.sbatch kept the discovery record and continued after a failed /v1/models probe (a warning only), and a vLLM pid can be
alive while its endpoint answers nothing — so with A serving live targets and B's record present while B's endpoint was down, `retire-member A`
withdrew A and the targets had no attacker. Now a member PROVES its health with a FRESH lease it writes itself (<pool>/<job>.health: ok, utc,
lease_s — right after its record and on every tick after its probe), the retirement transaction counts a survivor only with a fresh ok lease
(fail closed: none / stale / ok=false is no server) and honours the <job>.UNHEALTHY tombstone a member leaves when its probe fails
F22_POOL_HEALTH_MAX_FAILS times in a row and it withdraws at the source."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402
from test_attacker_pool_lifecycle import _call, _world  # noqa: E402
from test_pool_member_lifecycle_funnel import GROUP  # noqa: E402
from test_pool_retire_funnel import _bind, _lease, _world2  # noqa: E402

SB = REPO / "scripts" / "sdfbase" / "full22" / "attacker_pool.sbatch"


def test_retire_member_refuses_when_the_only_other_live_record_has_no_fresh_ok_lease_and_allows_it_once_it_does(tmp_path):
    w, (j1, j2), (lw1, lw2), (t1, t2), root, pool = _world2(tmp_path)
    _bind(root, t1, "http://n0:8707"); _bind(root, t2, "http://n0:8707")
    # B's job is live on sacct and its record stands, but its last probe FAILED (lease ok=false): B is no server — A is the last one, refused
    _lease(pool, j2, "http://n1:8708", ok=False)
    r = w.run("retire-member", "--group", GROUP, "--job", j1)
    assert r.returncode == 3 and "LAST serving member" in r.stderr and f"{j2}: health probe failed (lease ok=false)" in (r.stdout + r.stderr), r.stdout + r.stderr
    assert not (pool / f"{j1}.RETIRE").exists()
    # B's lease is STALE (older than its own lease_s): the same refusal, naming the staleness
    _lease(pool, j2, "http://n1:8708", ok=True, age_s=600, lease_s=180)
    r = w.run("retire-member", "--group", GROUP, "--job", j1)
    assert r.returncode == 3 and "health lease stale" in (r.stdout + r.stderr) and not (pool / f"{j1}.RETIRE").exists(), r.stdout + r.stderr
    # B has no lease at all (a member from before the contract, or one that never wrote it): fail closed
    (pool / f"{j2}.health").unlink()
    r = w.run("retire-member", "--group", GROUP, "--job", j1)
    assert r.returncode == 3 and f"{j2}: no health lease" in (r.stdout + r.stderr) and not (pool / f"{j1}.RETIRE").exists(), r.stdout + r.stderr
    # B withdrew itself as UNHEALTHY (its tombstone stands even if a stale record lingers): no server either
    _lease(pool, j2, "http://n1:8708"); (pool / f"{j2}.UNHEALTHY").write_text(json.dumps({"job_id": j2, "why": "health check failed 3 times in a row"}))
    r = w.run("retire-member", "--group", GROUP, "--job", j1)
    assert r.returncode == 3 and f"withdrew as unhealthy ['{j2}']" in (r.stdout + r.stderr) and not (pool / f"{j1}.RETIRE").exists(), r.stdout + r.stderr
    (pool / f"{j2}.UNHEALTHY").unlink()
    # B's lease is fresh and ok: A may retire; B is the survivor
    r = w.run("retire-member", "--group", GROUP, "--job", j1)
    assert r.returncode == 0 and "marked RETIRING" in (r.stdout + r.stderr) and (pool / f"{j1}.RETIRE").exists(), r.stdout + r.stderr
    assert json.loads((pool / f"{j1}.RETIRE").read_text())["survivors"] == [j2]


def test_the_member_writes_its_lease_withdraws_itself_after_repeated_failed_probes_and_the_sbatch_runs_both(tmp_path):
    w = _world(tmp_path); pool = w["root"] / "attackers"
    r = _call(w, "f22_pool_health_write", str(pool), "4242", "http://n0:8707", "1", "180"); assert r.returncode == 0, r.stderr
    h = json.loads((pool / "4242.health").read_text()); assert h["ok"] is True and h["lease_s"] == 180 and h["url"] == "http://n0:8707" and h["utc"].endswith("Z")
    r = _call(w, "f22_pool_health_write", str(pool), "4242", "http://n0:8707", "0", "180"); assert r.returncode == 0
    assert json.loads((pool / "4242.health").read_text())["ok"] is False and not list(pool.glob(".4242.health.tmp.*"))    # rewritten atomically, no temp left
    (pool / "4242.json").write_text("{}"); (pool / "4242.pending").write_text("{}")
    r = _call(w, "f22_pool_member_unhealthy", str(pool), "4242", "http://n0:8707", 'health check failed 3 "times" in a row'); assert r.returncode == 0, r.stderr
    assert not (pool / "4242.json").exists() and not (pool / "4242.pending").exists()
    t = json.loads((pool / "4242.UNHEALTHY").read_text()); assert t["job_id"] == "4242" and t["why"] == "health check failed 3 times in a row"
    sb = SB.read_text()
    assert sb.index("attacker LIVE") < sb.index('f22_pool_health_write "$POOL_DIR" "$SLURM_JOB_ID" "$URL" 1 "$HEALTH_LEASE_S"') < sb.index("while :; do")   # the lease right after the record
    loop = sb[sb.index("while :; do"):]
    assert loop.count("f22_pool_health_write") == 2 and 'HEALTH_FAILS=$((HEALTH_FAILS + 1))' in loop and '"$HEALTH_FAILS" -ge "$HEALTH_MAX_FAILS"' in loop \
        and "f22_pool_member_unhealthy" in loop and '"reason":"unhealthy"' in loop and "exit 3" in loop.split("f22_pool_member_unhealthy", 1)[1].split("fi", 1)[0]
    assert "F22_POOL_HEALTH_LEASE_S:-180" in sb and "F22_POOL_HEALTH_MAX_FAILS:-3" in sb
    assert "WARNING: health check failed on $URL (kept; the drivers re-check before use)" not in sb                        # the warning-only path is gone


def test_a_probe_passes_only_on_a_2xx_naming_the_served_model_and_a_503_with_exit_0_withdraws_the_member(tmp_path):
    """31d-fix-4: `curl -s` exits 0 on a 503, so a member whose API answered 5xx renewed ok=true every tick and
    the retire walk counted it as the survivor while withdrawing the working member. The probe now passes only on a 2xx (curl --fail) whose body
    names the served model. Executed, never string-matched: the sbatch's own probe function under a stub curl whose 503 body even NAMES the
    model (only --fail can refuse it) and whose 200 can name another model; then the sbatch's own health loop over the real lease functions."""
    from test_attacker_pool_lifecycle import LIFE
    w = _world(tmp_path); pool = w["root"] / "attackers"; serve = tmp_path / "serve"; serve.mkdir()
    sb = SB.read_text()
    fn = sb[sb.index("pool_probe_model_ok(){"):sb.index("\npool_probe_ok || f22_fatal") + 1]                              # both probe functions, as the sbatch defines them
    loop = sb[sb.index("while :; do"):sb.rindex("done") + len("done")]
    assert "if pool_probe_ok; then" in loop and "curl -sf -m 20" in fn and "curl -s -m 20" not in sb and sb.index("pool_probe_ok(){") < sb.index("pool_probe_ok || f22_fatal")   # startup and every tick probe through it
    assert 'grep -q -F "$SERVED_NAME"' not in sb and "| pool_probe_model_ok" in fn                                                      # 31d-fix-5: no substring match anywhere; the body is parsed
    stub = tmp_path / "stubbin"; stub.mkdir(); mode = tmp_path / "mode"; clog = tmp_path / "curl.log"
    (stub / "curl").write_text(textwrap.dedent("""\
        #!/bin/bash
        # the member's API as the probe sees it: MODE 503 answers a 5xx whose body NAMES the model (exit 0 unless --fail is asked, as curl does); wrongmodel a 2xx naming another model; ok a 2xx naming the model
        mode=$(cat "$CURL_MODE_FILE"); fail=0
        for a in "$@"; do case "$a" in -f|--fail|--fail-with-body|-sf|-fs|-sfS|-fsS) fail=1;; esac; done
        echo "CURL $*" >> "$CURL_LOG"
        case "$mode" in
          503) [ "$fail" = 1 ] && exit 22; printf '{"error":"503 Service Unavailable","model":"%s"}' "$SERVED_NAME_STUB"; exit 0;;
          wrongmodel) printf '{"object":"list","data":[{"id":"some-other-model"}]}'; exit 0;;
          superprefix) printf '{"object":"list","data":[{"id":"huihui-ai/%s"}]}' "$SERVED_NAME_STUB"; exit 0;;
          supersuffix) printf '{"object":"list","data":[{"id":"%s-abliterated-v2"}]}' "$SERVED_NAME_STUB"; exit 0;;
          among) printf '{"object":"list","data":[{"id":"some-other-model"},{"id":"%s"},{"id":"third-model"}]}' "$SERVED_NAME_STUB"; exit 0;;
          notjson) printf '<html>Service Unavailable: %s</html>' "$SERVED_NAME_STUB"; exit 0;;
          ok) printf '{"object":"list","data":[{"id":"%s"}]}' "$SERVED_NAME_STUB"; exit 0;;
        esac
        exit 7
        """)); (stub / "curl").chmod(0o755)
    env = {**w["env"], "PATH": f"{stub}:{w['env']['PATH']}", "CURL_MODE_FILE": str(mode), "CURL_LOG": str(clog), "SERVED_NAME_STUB": "attacker-model"}
    pre = textwrap.dedent(f"""
        set -u
        source {LIFE}
        f22_log(){{ echo "[log] $*"; }}
        f22_stop_requested(){{ return 1; }}
        f22_pool_retire_requested(){{ return 1; }}
        f22_pool_consumers(){{ :; }}
        f22_pool_member_verdict(){{ echo keep; echo stub >&2; }}
        sleep(){{ :; }}
        URL=http://TESTHOST:8707; SERVED_NAME=attacker-model; POOL_DIR={pool}; SLURM_JOB_ID=4242; WAVE_ROOT={w['root']}; SERVE_DIR={serve}; VLLM_PID=$$
        REC={pool}/4242.json; PENDING={pool}/4242.pending
        HEALTH_FAILS=0; HEALTH_LEASE_S=180; HEALTH_MAX_FAILS=3; LAST_VERDICT_WHY=""
    """) + fn
    def run(body):
        return subprocess.run(["bash", "-c", pre + "\n" + body], capture_output=True, text=True, env=env, timeout=120)
    mode.write_text("ok"); r = run("pool_probe_ok; echo rc=$?"); assert "rc=0" in r.stdout, r.stdout + r.stderr
    mode.write_text("503"); r = run("pool_probe_ok; echo rc=$?"); assert "rc=1" in r.stdout, r.stdout + r.stderr                    # the body names the model: --fail decides
    assert "CURL -sf -m 20 http://TESTHOST:8707/v1/models" in clog.read_text()
    mode.write_text("wrongmodel"); r = run("pool_probe_ok; echo rc=$?"); assert "rc=1" in r.stdout, r.stdout + r.stderr             # a 2xx naming another model is no server
    # the id is matched EXACTLY against data[].id, never as a substring of the body — a model whose id
    # merely CONTAINS the expected one (a prefix or a suffix around it) is another model; the exact id among others passes; a non-JSON 2xx fails
    for m_, want in (("superprefix", "rc=1"), ("supersuffix", "rc=1"), ("among", "rc=0"), ("notjson", "rc=1")):
        mode.write_text(m_); r = run("pool_probe_ok; echo rc=$?"); assert want in r.stdout, (m_, r.stdout + r.stderr)
    # the loop: the API answers 503 (exit 0 without --fail) while the job and the vLLM pid stay live — three ticks, each lease ok=false, then withdrawn
    (pool / "4242.json").write_text("{}"); (pool / "4242.pending").write_text("{}")
    mode.write_text("503"); r = run(loop)
    assert r.returncode == 3 and "SELF_EXIT (unhealthy): 3 failed probes in a row" in r.stdout and r.stdout.count("WARNING: health probe failed on http://TESTHOST:8707") == 3, r.stdout + r.stderr
    assert json.loads((pool / "4242.health").read_text())["ok"] is False and json.loads((pool / "4242.UNHEALTHY").read_text())["why"] == "health check failed 3 times in a row" \
        and not (pool / "4242.json").exists() and not (pool / "4242.pending").exists()
    # a passing probe resets the count: ok, ok, 503, ok → the lease is ok again and nothing is withdrawn (one tick each; the loop is left by STOP)
    (pool / "4242.UNHEALTHY").unlink(); (pool / "4242.json").write_text("{}")
    seq = tmp_path / "seq"; seq.write_text("ok\n503\nok\nok\n")
    (stub / "curl").write_text((stub / "curl").read_text().replace('mode=$(cat "$CURL_MODE_FILE")', 'mode=$(head -n 1 "$CURL_SEQ_FILE"); tail -n +2 "$CURL_SEQ_FILE" > "$CURL_SEQ_FILE.n"; mv "$CURL_SEQ_FILE.n" "$CURL_SEQ_FILE"; [ -n "$mode" ] || { touch "$POOL_DIR_STUB/STOP"; mode=ok; }'))
    env2 = {**env, "CURL_SEQ_FILE": str(seq), "POOL_DIR_STUB": str(pool)}
    r = subprocess.run(["bash", "-c", pre + "\n" + loop], capture_output=True, text=True, env=env2, timeout=120)
    assert r.returncode == 0 and "STOP requested; leaving" in r.stdout and r.stdout.count("WARNING: health probe failed") == 1 and "SELF_EXIT" not in r.stdout, r.stdout + r.stderr
    assert json.loads((pool / "4242.health").read_text())["ok"] is True and not (pool / "4242.UNHEALTHY").exists()


def _path_without(path: str, name: str, farm: Path) -> str:
    """A one-directory PATH holding a symlink to every command of ``path``'s directories (first dir wins, as PATH does) except ``name`` —
    a host 'without' that command, whatever the real host carries."""
    import os
    farm.mkdir(parents=True, exist_ok=True); seen = set()
    for d in path.split(os.pathsep):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for n in names:
            if n in seen or n == name:
                continue
            seen.add(n)
            try:
                os.symlink(os.path.join(d, n), farm / n)
            except OSError:
                pass
    return str(farm)


def test_the_member_samples_its_queue_and_device_every_15_s_inside_the_tick_into_an_append_only_sidecar(tmp_path):
    """The member samples the engine's queue (vLLM /metrics: running / waiting + the latency histogram sums and
    counts) and the device (nvidia-smi over Slurm's own device list) every 15 s inside the 60 s tick, one JSON line per sample appended to
    <pool>/<job>.samples.jsonl beside the health lease; a member whose engine answers no metrics or whose node has no nvidia-smi appends a null
    row and never fails. Executed: the sbatch's own sampler and its own loop, /metrics served by a local HTTP server (the sampler reads it with
    urllib, never through curl — the probe's curl path and its stubs stay the health check's alone), nvidia-smi a stub."""
    import http.server
    import threading
    from test_attacker_pool_lifecycle import LIFE
    w = _world(tmp_path); pool = w["root"] / "attackers"; serve = tmp_path / "serve"; serve.mkdir()
    sb = SB.read_text()
    fn = sb[sb.index("pool_probe_model_ok(){"):sb.index("\npool_probe_ok || f22_fatal") + 1]
    loop = sb[sb.index("while :; do"):sb.rindex("done") + len("done")]
    assert "pool_sample(){" in fn and loop.count("pool_sample") == 1 and 'for _ in 1 2 3 4; do pool_sample; sleep "${F22_POOL_SAMPLE_EVERY_S:-15}"; done' in loop and "\n  sleep 60\n" not in loop
    assert 'nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES:-0}"' in fn   # Slurm's own device list
    body = fn[fn.index("pool_sample(){"):]
    code = "\n".join(l.split("#", 1)[0] for l in body.splitlines())                                   # comments off: the sampler's CODE never calls curl
    assert "urllib.request.urlopen" in code and "curl" not in code
    metrics = ("# HELP vllm:num_requests_running x\n# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running{model_name=\"m\"} 3.0\n"
               "vllm:num_requests_waiting{model_name=\"m\"} 2.0\nvllm:request_queue_time_seconds_sum{model_name=\"m\"} 12.5\n"
               "vllm:request_queue_time_seconds_count{model_name=\"m\"} 10.0\nvllm:e2e_request_latency_seconds_sum{model_name=\"m\"} 400.0\n"
               "vllm:e2e_request_latency_seconds_count{model_name=\"m\"} 10.0\nvllm:time_to_first_token_seconds_bucket{le=\"0.1\",model_name=\"m\"} 4.0\nother_metric 1.0\n")
    seen = []; body = {"text": metrics}

    class Metrics(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            if self.path != "/metrics" or body["text"] is None:
                self.send_response(503); self.end_headers(); return
            data = body["text"].encode()
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Metrics); port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    stub = tmp_path / "stubbin"; stub.mkdir(); clog = tmp_path / "calls.log"
    (stub / "curl").write_text('#!/bin/bash\necho "CURL $*" >> "$CALL_LOG"\nprintf \'{"object":"list","data":[{"id":"%s"}]}\' "$SERVED_NAME_STUB"; [ -n "${POOL_DIR_STUB:-}" ] && touch "$POOL_DIR_STUB/STOP"; exit 0\n'); (stub / "curl").chmod(0o755)
    (stub / "nvidia-smi").write_text('#!/bin/bash\necho "NVSMI $*" >> "$CALL_LOG"; echo "0, 87, 41, 120000, 143000"\n'); (stub / "nvidia-smi").chmod(0o755)
    env = {**w["env"], "PATH": f"{stub}:{w['env']['PATH']}", "CALL_LOG": str(clog), "SERVED_NAME_STUB": "attacker-model", "CUDA_VISIBLE_DEVICES": "3"}
    pre = textwrap.dedent(f"""
        set -u
        source {LIFE}
        f22_log(){{ echo "[log] $*"; }}
        f22_stop_requested(){{ return 1; }}
        f22_pool_retire_requested(){{ return 1; }}
        f22_pool_consumers(){{ :; }}
        f22_pool_member_verdict(){{ echo keep; echo stub >&2; }}
        sleep(){{ :; }}
        URL=http://127.0.0.1:{port}; SERVED_NAME=attacker-model; POOL_DIR={pool}; SLURM_JOB_ID=4242; WAVE_ROOT={w['root']}; SERVE_DIR={serve}; VLLM_PID=$$
        REC={pool}/4242.json; PENDING={pool}/4242.pending
        HEALTH_FAILS=0; HEALTH_LEASE_S=180; HEALTH_MAX_FAILS=3; LAST_VERDICT_WHY=""
    """) + fn
    r = subprocess.run(["bash", "-c", pre + "\npool_sample; pool_sample; echo rc=$?"], capture_output=True, text=True, env=env, timeout=120)
    assert "rc=0" in r.stdout, r.stdout + r.stderr
    rows = [json.loads(l) for l in (pool / "4242.samples.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["job_id"] == "4242" and rows[0]["url"] == f"http://127.0.0.1:{port}" and rows[0]["utc"].endswith("Z") and rows[0]["metrics_ok"] is True
    assert rows[0]["running"] == 3.0 and rows[0]["waiting"] == 2.0
    assert rows[0]["histograms"] == {"vllm:request_queue_time_seconds_sum": 12.5, "vllm:request_queue_time_seconds_count": 10.0,
                                     "vllm:e2e_request_latency_seconds_sum": 400.0, "vllm:e2e_request_latency_seconds_count": 10.0}   # buckets and foreign metrics are not kept
    assert rows[0]["gpu"] == [{"index": 0, "util_gpu_pct": 87.0, "util_mem_pct": 41.0, "mem_used_mib": 120000.0, "mem_total_mib": 143000.0}]
    assert seen == ["/metrics", "/metrics"] and "CURL" not in clog.read_text()                                                             # read by urllib; curl untouched
    assert "NVSMI --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits -i 3" in clog.read_text()
    # the engine answers 503 and the node has no nvidia-smi: a null row, rc 0 — nothing here may fail the member. 'No nvidia-smi' is a
    # PATH without one (every other command of the host's PATH, linked into a farm): unlinking the stub alone left the host's real
    # nvidia-smi in reach on the cluster gate's GPU node, which answered for device 3 (2026-10-08: one INTRODUCED node)
    (stub / "nvidia-smi").unlink(); body["text"] = None
    env_nosmi = {**env, "PATH": f"{stub}:{_path_without(w['env']['PATH'], 'nvidia-smi', tmp_path / 'nosmi')}"}
    r = subprocess.run(["bash", "-c", pre + "\npool_sample; echo rc=$?"], capture_output=True, text=True, env=env_nosmi, timeout=120)
    rows = [json.loads(l) for l in (pool / "4242.samples.jsonl").read_text().splitlines()]
    assert "rc=0" in r.stdout and len(rows) == 3 and rows[2]["metrics_ok"] is False and rows[2]["running"] is None and rows[2]["waiting"] is None and rows[2]["gpu"] == [] and rows[2]["histograms"] == {}, r.stdout + r.stderr
    # the loop: four samples per tick; the tick's probe and lease unchanged (the probe's stub plants STOP, so the loop leaves after one tick)
    (pool / "4242.json").write_text("{}"); (pool / "4242.pending").write_text("{}"); (pool / "4242.samples.jsonl").unlink()
    body["text"] = "vllm:num_requests_running 1.0\nvllm:num_requests_waiting 0.0\n"
    r = subprocess.run(["bash", "-c", pre + "\n" + loop], capture_output=True, text=True, env={**env, "POOL_DIR_STUB": str(pool)}, timeout=120)
    assert r.returncode == 0 and "STOP requested; leaving" in r.stdout and "WARNING" not in r.stdout, r.stdout + r.stderr
    rows = [json.loads(l) for l in (pool / "4242.samples.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and all(x["running"] == 1.0 and x["waiting"] == 0.0 and x["metrics_ok"] is True for x in rows), rows
    assert json.loads((pool / "4242.health").read_text())["ok"] is True and clog.read_text().count("CURL") == 1                              # one probe, four samples
    srv.shutdown()
