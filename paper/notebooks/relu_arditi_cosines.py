"""Layer-12 cosine geometry of the ReLU steers against the Arditi refusal direction -- the sidecar behind the appendix figure
`figA_relu_arditi_cosines` (the user, 2026-09-25).

Run from the repository root with an interpreter that has torch (`uv sync --frozen --extra cpu` builds one):
    .venv/bin/python paper/notebooks/relu_arditi_cosines.py
It writes `paper/notebooks/data/relu_arditi_cosines.json`, which the notebook reads and BINDS to the registry: every entry
carries the sha256 of the object file it was read from, which is the registry's `object_id`, so the notebook can check
that each row is the object the registry calls by that label and seed, and refuse a stale or mislabelled sidecar.
The objects are named here BY THAT ID and resolved to the committed paper set through `objects/MANIFEST.json`
(`objects/<fit model_id>/<object_id>/object.pt`); the two directions that are not committed are each read from a variable
that names their `object.pt`: the instruct-direct constitution ReLU (not a paper object, a reference cosine only) from
`$SISTEER_COSINE_INSTRUCT_CONSTITUTION_RELU`, and the safety-distilled ReLU, which `objects/WITHHELD.json` withholds from this
repository, from `$SISTEER_COSINE_SAFETY_RELU`. With the variable unset the script refuses by name, and for a withheld object
with the reason the list records. Every file's sha256 is checked against the id it is named by. The
recorded `path` is repository-relative for a committed object and the variable's name (`$…`) for the fallback, so the
sidecar never carries a machine's directory layout.

What is compared, all at decoder block 12 (zero-based, resid_post -- the layer the Arditi direction was extracted at and one
of the 32 the ReLU steers are fit on):
  * the Arditi refusal direction: `vectors["12"]` of an `arditi_allpos` rung object. The stored vector is c times the unit
    direction (the c = 0.15 and c = -0.5 rungs are exactly antiparallel), so the POSITIVE-c rung is the reference and the
    cosine is against the unit direction "toward refusal".
  * a ReLU steer's READ direction: `gate_w["12"]`, the probe whose ReLU(w . h + b) gates the write, and its WRITE direction:
    `directions["12"]`, the vector added to the residual stream (PSRObject: h + g(h) d).
  * for each fitted family with several seeds: every seed's read and write direction against the reference, and each pair
    of seeds against each other (read with read, write with write). The single instruct-direct fits enter as reference
    cosines only. The paired-write difference ReLU is a composition (u_pos - u_neg on the shared pre-write state) with no
    single direction of its own and is not a row here.
The random-direction reference is the sd of the cosine of two independent random unit vectors in d dimensions, 1 / sqrt(d)."""
from __future__ import annotations
import glob, hashlib, itertools, json, math, os, subprocess, sys, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                                 # paper/notebooks -> the repository root
OUT = os.path.join(HERE, "data", "relu_arditi_cosines.json")
LAYER = "12"
OBJECTS_ROOT = os.path.join(REPO, "objects")
MANIFEST = os.path.join(OBJECTS_ROOT, "MANIFEST.json")
WITHHELD = os.path.join(OBJECTS_ROOT, "WITHHELD.json")                       # paper objects this repository does not ship
INSTRUCT_CONSTITUTION_RELU_ENV = "SISTEER_COSINE_INSTRUCT_CONSTITUTION_RELU"  # the one direction the paper set does not carry
SAFETY_RELU_ENV = "SISTEER_COSINE_SAFETY_RELU"                               # a paper object objects/WITHHELD.json withholds
# the positive-c Arditi rung (`arditi_allpos`, c = 0.15): the unit direction toward refusal
REFERENCE_ID = "a2df64624b36be96c9ca50b36907b132c61382ea81ca3d3841e87c0fe012cff5"
OBJECTS = [  # (paper_label, seed, fit checkpoint, registry object_id = sha256(object.pt), fallback env var when not committed)
    ("control_relu", "s1", "qwen35_4b_base", "f65425b851ab446b329fa93631462dd4f31fb16e11955867ef902dffd635c2cd", None),
    ("control_relu", "s2", "qwen35_4b_base", "900fbabb6d56a61f37acf71a28482f8dd843a6f27809f12f09ca38a1328bbecb", None),
    ("control_relu", "s3", "qwen35_4b_base", "c5686c7316b7f79ba31edeea4ecf5d3b237724e399cafb41295478b457471c34", None),
    ("constitution_relu", "s1", "qwen35_4b_base", "7b87767d91bd9787d22a23a80a5e78c53d16b506cc25e4d9483f74e88238fda1", None),
    ("constitution_relu", "s2", "qwen35_4b_base", "2d43a58e55e674d3e33c6cbbaf7bb0ad9b06021d308b9e924213acf7c0609fd8", None),
    ("constitution_relu", "s3", "qwen35_4b_base", "cc1ec068380ed8fe51de1e91e8b8fe6b32c95fa9a00bcf381a536007c075bf09", None),
    ("constitution_relu", "", "qwen35_4b_it", "0a69aab85e1ead1e3b38d377e197fb03520c5ae7ca1930185ce0990d46bb3482", INSTRUCT_CONSTITUTION_RELU_ENV),
    ("safety_relu", "", "qwen35_4b_it", "75625f59809498899db9efcd2b19e9ce8c2f95593e7c7c1a109a03e29fe5b5c8", SAFETY_RELU_ENV),
]


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def committed_object(object_id: str, fallback_env: str | None = None) -> tuple[str, str]:
    """(file, recorded path) of a registry object_id: the committed `objects/<fit model_id>/<object_id>/object.pt` resolved
    through objects/MANIFEST.json (`dir`), recorded repository-relative; else the file `$<fallback_env>` names, recorded as the
    variable (`$NAME`), never its value; else a refusal by name. The bytes are hashed and must equal the id they are named by."""
    rows = {o["object_id"]: o for o in json.load(open(MANIFEST))["objects"]}
    row = rows.get(object_id)
    if row is not None and row.get("dir"):
        path = os.path.join(OBJECTS_ROOT, row["dir"], "object.pt"); recorded = os.path.relpath(path, REPO)
    elif fallback_env and os.environ.get(fallback_env):
        path = os.environ[fallback_env]; recorded = f"${fallback_env}"
    else:
        held = ({w["object_id"]: w for w in json.load(open(WITHHELD))["objects"]}.get(object_id)
                if os.path.isfile(WITHHELD) else None)
        why = (f"object {object_id} ({held['model_id']} {held['paper_label']}) is withheld from this repository: {held['reason']} "
               f"(objects/WITHHELD.json)" if held else f"object {object_id[:12]}… is not in objects/MANIFEST.json")
        raise SystemExit(f"{why}, and ${fallback_env or '<no fallback variable>'} does not name its object.pt")
    got = sha(path)
    if got != object_id:
        raise SystemExit(f"{path}: sha256 {got[:12]}… != the object_id {object_id[:12]}… it is named by")
    return path, recorded


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float(), b.float(), dim=0))


def main() -> None:
    reference, reference_recorded = committed_object(REFERENCE_ID)
    ref_obj = torch.load(reference, map_location="cpu", weights_only=False)
    ref = ref_obj["vectors"][LAYER]
    d = int(ref.numel())
    entries, dirs = [], {}
    for label, seed, fit_model, object_id, fallback_env in OBJECTS:
        path, recorded = committed_object(object_id, fallback_env)
        o = torch.load(path, map_location="cpu", weights_only=False)
        assert o["kind"] == "psr", (path, o["kind"])
        r, w = o["gate_w"][LAYER], o["directions"][LAYER]
        dirs[(label, seed)] = (r, w)
        entries.append(dict(object_label=label, seed=seed, fit_model_id=fit_model, object_id=object_id, path=recorded,
                            read_vs_ref=cos(r, ref), write_vs_ref=cos(w, ref), read_vs_write=cos(r, w),
                            read_norm=float(r.norm()), write_norm=float(w.norm()), gate_b=float(o["gate_b"][LAYER].item())))
    pairs = []
    for label in sorted({l for l, s in dirs if s}):
        seeds = sorted(s for l, s in dirs if l == label and s)
        for a, b in itertools.combinations(seeds, 2):
            pairs.append(dict(object_label=label, seed_a=a, seed_b=b, read=cos(dirs[(label, a)][0], dirs[(label, b)][0]),
                              write=cos(dirs[(label, a)][1], dirs[(label, b)][1])))
    # ACROSS the two seeded objects (the user, 2026-09-25): every control seed against every constitution seed, read with read and
    # write with write -- do the two recipes find the same direction, or each its own?
    cross = []
    seeded = sorted({l for l, s in dirs if s})
    for la, lb in itertools.combinations(seeded, 2):
        for a in sorted(s for l, s in dirs if l == la and s):
            for b in sorted(s for l, s in dirs if l == lb and s):
                cross.append(dict(object_a=la, seed_a=a, object_b=lb, seed_b=b, read=cos(dirs[(la, a)][0], dirs[(lb, b)][0]),
                                  write=cos(dirs[(la, a)][1], dirs[(lb, b)][1])))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:                                                         # noqa: BLE001 - provenance only
        commit = None
    out = dict(layer=int(LAYER), model_id="qwen35_4b_it", d=d, random_sd=1.0 / math.sqrt(d),
               reference=dict(object_label="arditi_allpos", object_id=REFERENCE_ID, path=reference_recorded, norm=float(ref.norm()),
                              note="the c = 0.15 rung's stored vector; cosines are against its unit direction (toward refusal)"),
               entries=entries, pairs=pairs, cross_pairs=cross,
               generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script=os.path.relpath(__file__, REPO),
               script_sha256=sha(__file__), git_commit=commit, torch=torch.__version__)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}: {len(entries)} directions, {len(pairs)} seed pairs, {len(cross)} cross-object pairs, d = {d}, random sd {out['random_sd']:.4f}")


if __name__ == "__main__":
    main()
