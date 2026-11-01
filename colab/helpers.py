"""Helpers for colab/eval.ipynb: progress logs, streamed commands, GPU checks and the model server.

Every log line starts with the time and the step, so the output always shows where the run is:

    [14:02:11 | step 4/7] converting to BF16, then Q8_0 ...

Long commands stream their output into the cell as it happens. Very chatty lines (one per tensor
during conversion) are hidden, and a "still working" line appears every 15 s instead. Progress bars
redrawn in place (downloads) are shown at most every 10 s.
"""

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

TOTAL_STEPS = 7
NOTEBOOK_STARTED = time.time()
TIMINGS = {}                    # step title -> seconds, saved in run_info.json
_step = {"n": 0, "title": "", "started": time.time()}


class StopHere(Exception):
    """Raised to stop "Run all" with a message that says what went wrong and what to do."""


def fmt(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} s"
    m, s = divmod(seconds, 60)
    return f"{m} min {s:02d} s" if m < 60 else f"{m // 60} h {m % 60:02d} min"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} | step {_step['n']}/{TOTAL_STEPS}] {msg}", flush=True)


def step(n, title, purpose):
    """Start a step: a banner with what the step does and how far into the notebook we are."""
    _step.update(n=n, title=title, started=time.time())
    print("\n" + "=" * 80)
    print(f"STEP {n}/{TOTAL_STEPS}: {title}")
    for line in purpose.strip().splitlines():
        print(f"  {line.strip()}")
    print(f"  (started {time.strftime('%H:%M:%S')}, {fmt(time.time() - NOTEBOOK_STARTED)} after the notebook started)")
    print("=" * 80, flush=True)


def done(summary):
    took = time.time() - _step["started"]
    TIMINGS[f"{_step['n']}. {_step['title']}"] = round(took)
    log(f"DONE in {fmt(took)}: {summary}")
    if _step["n"] < TOTAL_STEPS:
        print(f"  -> next: step {_step['n'] + 1}/{TOTAL_STEPS}", flush=True)


def fail(problem, fix, details=""):
    msg = f"\n\nSTEP {_step['n']}/{TOTAL_STEPS} FAILED: {problem}\n"
    if details:
        msg += f"\nLast output:\n{details}\n"
    msg += f"\nWhat to do: {fix}\n"
    raise StopHere(msg)


def run(cmd, *, cwd=None, env=None, quiet=None, check=True, on_fail="read the output above", mask=()):
    """Run a command, streaming its output into the cell. Returns (exit code, output lines).

    quiet: regex for chatty lines to hide. mask: strings (tokens) never to print.
    """
    proc = subprocess.Popen(cmd, cwd=cwd, env={**os.environ, **(env or {})},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    quiet_re = re.compile(quiet) if quiet else None
    lines, buf = [], b""
    state = {"hidden": 0, "beat": time.time(), "progress": 0.0}

    def emit(text, progress):
        for secret in mask:
            if secret:
                text = text.replace(secret, "***")
        if not text.strip():
            return
        if progress:                                   # redrawn in place: show now and then
            if time.time() - state["progress"] < 10:
                return
            state["progress"] = time.time()
        else:
            lines.append(text)
            if quiet_re and quiet_re.search(text):
                state["hidden"] += 1
                if time.time() - state["beat"] > 15:
                    log(f"... still working ({state['hidden']} detail lines hidden so far)")
                    state["beat"] = time.time()
                return
        print(f"    {text}", flush=True)

    while True:
        chunk = proc.stdout.read1(65536)
        if not chunk:
            break
        buf += chunk
        while (m := re.search(rb"\r\n|\r|\n", buf)):
            if m.group() == b"\r" and m.end() == len(buf):
                break                                   # maybe the first half of \r\n: wait for more
            emit(buf[: m.start()].decode(errors="replace"), progress=m.group() == b"\r")
            buf = buf[m.end():]
    if buf:
        emit(buf.decode(errors="replace"), progress=False)
    code = proc.wait()
    if check and code != 0:
        fail(f"{' '.join(map(str, cmd[:3]))} ... exited with code {code}", on_fail,
             "\n".join(f"    {line}" for line in lines[-15:]))
    return code, lines


def gpu_info():
    """The GPU Colab gave us, or None. Includes the highest CUDA version its driver supports."""
    if not shutil.which("nvidia-smi"):
        return None
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                        "--format=csv,noheader"], capture_output=True, text=True)
    if q.returncode or not q.stdout.strip():
        return None
    name, driver, memory, cap = [x.strip() for x in q.stdout.splitlines()[0].split(",")]
    banner = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", banner)
    return {"name": name, "driver": driver, "memory": memory, "compute_cap": cap,
            "driver_cuda": cuda.group(1) if cuda else "unknown"}


def system_info(disk_path="/content"):
    os_name = "unknown"
    if os.path.exists("/etc/os-release"):
        os_name = dict(line.rstrip().split("=", 1) for line in open("/etc/os-release") if "=" in line) \
            .get("PRETTY_NAME", "unknown").strip('"')
    ram_gb = None
    if os.path.exists("/proc/meminfo"):
        kb = int(re.search(r"MemTotal:\s+(\d+)", open("/proc/meminfo").read()).group(1))
        ram_gb = round(kb / 1024 ** 2, 1)
    disk = shutil.disk_usage(disk_path if os.path.exists(disk_path) else ".")
    return {"os": os_name, "cpus": os.cpu_count(), "ram_gb": ram_gb, "disk_free_gb": round(disk.free / 1e9)}


def cuda_devices(llama_dir):
    """Devices llama.cpp can use, e.g. ['CUDA0: Tesla T4 (15095 MiB, 14992 MiB free)']."""
    if not os.path.exists(f"{llama_dir}/llama-server"):
        return [], [f"{llama_dir}/llama-server not found"]
    q = subprocess.run([f"{llama_dir}/llama-server", "--list-devices"], capture_output=True, text=True,
                       env={**os.environ, "LD_LIBRARY_PATH": f"{llama_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"})
    out = (q.stdout + q.stderr).splitlines()
    return [line.strip() for line in out if re.match(r"\s*CUDA\d+:", line)], out


def build_llamacpp_from_source(build, llama_dir, cache_dir, compute_cap):
    """Fallback when the prebuilt CUDA download can't use this GPU (driver too old): compile the pinned
    build with Colab's own CUDA toolkit, for this GPU only, and keep a copy in Drive for next time."""
    arch = compute_cap.replace(".", "")
    cached = f"{cache_dir}/llama.cpp-{build}-cuda-sm{arch}.tar.gz"
    shutil.rmtree(llama_dir, ignore_errors=True)
    os.makedirs(llama_dir)
    if os.path.exists(cached):
        log(f"found a compiled copy in Drive: {cached}")
        run(["tar", "-xzf", cached, "-C", llama_dir])
        return
    src = "/content/llama.cpp-src"
    log(f"compiling llama.cpp {build} for sm_{arch}: about 20-40 min on Colab's 2 CPU cores (first time only)")
    if not os.path.exists(src):
        # Full commit history (no --depth) so the build reports the right build number.
        run(["git", "clone", "--filter=blob:none", "--quiet", "https://github.com/ggml-org/llama.cpp.git", src])
    run(["git", "-C", src, "-c", "advice.detachedHead=false", "checkout", "--quiet", build])
    env = {"PATH": f"/usr/local/cuda/bin:{os.environ['PATH']}"}
    run(["cmake", "-S", src, "-B", f"{src}/build", "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
         "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF"], env=env, quiet=r"^--")
    run(["cmake", "--build", f"{src}/build", "--config", "Release", "-j", str(os.cpu_count() or 2),
         "--target", "llama-server", "llama-quantize"], env=env, quiet=r"^\[\s*\d+%\]")
    run(["bash", "-c", f'cp -a "{src}/build/bin/." "{llama_dir}/"'])
    os.makedirs(cache_dir, exist_ok=True)
    run(["tar", "-czf", cached, "-C", llama_dir, "."])
    log(f"saved the compiled copy to Drive: {cached}")


def kill_stray_servers():
    """A server left over from an earlier run (e.g. before a kernel restart) would block the port."""
    if subprocess.run(["pkill", "-f", "llama-server"], capture_output=True).returncode == 0:
        log("stopped a llama-server left over from an earlier run")
        time.sleep(2)


def free_port(preferred=8181):
    """A local port nothing listens on: `preferred` if free, else one the OS picks.
    (Colab's own runtime already listens on 8080, so the llama.cpp default can't be used there.)"""
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise OSError("no free local port")


def start_server(repo, model, quant, gguf, log_path, port=8080):
    """Start llama-server in the background with the shared settings (serve/run_local.sh)."""
    out = open(log_path, "w")
    return subprocess.Popen(["bash", "serve/run_local.sh", model, quant], cwd=repo,
                            env={**os.environ, "GGUF": gguf, "PORT": str(port)},
                            stdout=out, stderr=subprocess.STDOUT, start_new_session=True)


def tail(path, n=15):
    try:
        return [line.rstrip() for line in open(path, errors="replace").readlines()[-n:]]
    except OSError:
        return []


def wait_ready(proc, log_path, port=8080, timeout=900):
    """Wait until /health answers 200, printing what the server is doing every 15 s."""
    started, last = time.time(), 0.0
    while time.time() - started < timeout:
        if proc.poll() is not None:
            lines = tail(log_path)
            if any("couldn't bind" in line for line in lines):
                fail(f"llama-server couldn't use port {port}: another program is already listening on it",
                     "run this step again: it picks a free port each time", "\n".join(f"    {l}" for l in lines))
            fail(f"llama-server stopped (exit code {proc.returncode}) before it was ready",
                 "read the server log lines above; the full log is in the run folder",
                 "\n".join(f"    {line}" for line in lines))
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    log(f"server is ready ({fmt(time.time() - started)} to load)")
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass                                        # still starting (503 while loading)
        if time.time() - last > 15:
            last_line = (tail(log_path, 1) or ["(no output yet)"])[0][:150]
            log(f"loading the model... {fmt(time.time() - started)}; server says: {last_line}")
            last = time.time()
        time.sleep(3)
    fail(f"llama-server was not ready after {fmt(timeout)}", "read the server log; then rerun this step")


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    log("model server stopped (GPU memory is free again)")


def parse_result(lines, prefix="RESULT "):
    """The key=value pairs of the last line starting with prefix (serve/get_gguf.sh's summary line)."""
    for line in reversed(lines):
        if line.startswith(prefix):
            return dict(part.split("=", 1) for part in line[len(prefix):].split())
    return {}


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
