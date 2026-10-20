"""stores: a cluster_reads.yaml destination that spells neither ``host`` nor ``shared`` is named by a cluster's registry label and takes its
login host, its shared root and its repo mirror from the registry configuration's [[clusters]] entry of the same name: the mirror is the
entry's ``remote_src`` without ``/src``, else ``<shared>/repos/<the directory name of [repo] checkout>``. The join happens when the
destination is asked for (``Config.destination``), so a read of the components alone (a cluster read shard loading the staged file on a
node with no registry configuration, the coverage check) needs none; a destination whose label no entry names is refused by name then. A
destination that spells its own values keeps them."""
from __future__ import annotations

import pytest

from sisteer.registry import config as RC
from sisteer.registry import stores as ST

YAML = """schema: cluster_reads/2
written_utc: "2026-01-01T00:00:00Z"
destinations:
  label_one: {}
  label_two:
    stores_rel: registry/stores
    verify_cpus: 4
  local_dest:
    shared: /tmp/local-root
adapters: {}
components:
  comp:
    base: /tmp/somewhere
    authority: grading_host_copy
    note: a component
"""


def _yaml(tmp_path, text=YAML):
    y = tmp_path / "cluster_reads.yaml"
    y.write_text(text)
    return str(y)


def _root(tmp_path, clusters, checkout="/checkouts/foo"):
    reg = tmp_path / "reg"
    reg.mkdir(exist_ok=True)
    (reg / "config.toml").write_text(f'[repo]\ncheckout = "{checkout}"\n\n' + clusters)
    return str(reg)


def test_a_destination_takes_host_shared_and_its_mirror_from_the_clusters_entry_of_its_name(tmp_path):
    root = _root(tmp_path, '[[clusters]]\nname = "label_one"\nhost = "login-one"\nshared = "/shared/one"\nremote_src = "/shared/one/repos/clone/src"\n\n'
                           '[[clusters]]\nname = "label_two"\nhost = "login-two"\nshared = "/shared/{user}/two"\n')
    cfg = ST.load_config(_yaml(tmp_path), root=root)
    one, two, local = cfg.destination("label_one"), cfg.destination("label_two"), cfg.destination("local_dest")
    assert (one.host, one.shared, one.repo_rel, one.mirror_dir) == ("login-one", "/shared/one", "repos/clone", "/shared/one/repos/clone")
    user = RC.load(root)["registry"]["user"]
    assert (two.host, two.shared, two.repo_rel) == ("login-two", f"/shared/{user}/two", "repos/foo"), "no remote_src: repos/<checkout name>"
    assert two.verify_cpus == 4 and two.stores_dir == f"/shared/{user}/two/registry/stores", "the file's other keys are kept"
    assert local.is_local and local.shared == "/tmp/local-root", "a destination that spells its own values keeps them"
    assert cfg.declared() == ["label_one", "label_two", "local_dest"]


def test_a_remote_src_outside_the_shared_root_is_taken_whole(tmp_path):
    root = _root(tmp_path, '[[clusters]]\nname = "label_one"\nhost = "login-one"\nshared = "/shared/one"\nremote_src = "/elsewhere/clone/src/"\n')
    one = ST.load_config(_yaml(tmp_path), root=root).destination("label_one")
    assert one.repo_rel == "/elsewhere/clone" and one.mirror_dir == "/elsewhere/clone"


def test_a_destination_without_a_clusters_entry_is_refused_by_name_when_asked_for(tmp_path):
    root = _root(tmp_path, '[[clusters]]\nname = "label_one"\nhost = "login-one"\nshared = "/shared/one"\n')
    cfg = ST.load_config(_yaml(tmp_path), root=root)
    with pytest.raises(ST.StoresError, match=r"destination 'label_two' spells no `host` or `shared`, and the registry configuration .* names no "
                                             r"\[\[clusters\]\] entry 'label_two' \(declared: label_one\)"):
        cfg.destination("label_two")
    with pytest.raises(ST.StoresError, match=r"destination 'label_nine' is not declared"):
        cfg.destination("label_nine")
    with pytest.raises(ST.StoresError, match=r"--dest is required \(declared: label_one, label_two, local_dest\)"):
        cfg.destination(None)


def test_the_components_read_without_any_registry_configuration(tmp_path, monkeypatch):
    """The shard's case: the staged file on a node whose registry root holds no config.toml at all. Loading it reads the components and
    asks nothing of the clusters; only asking for a joined destination does."""
    empty = tmp_path / "no-registry"
    empty.mkdir()
    monkeypatch.setenv("REGISTRY_ROOT", str(empty))
    cfg = ST.load_config(_yaml(tmp_path))
    assert list(cfg.components) == ["comp"] and cfg.components["comp"].base == "/tmp/somewhere"
    assert cfg.destination("local_dest").shared == "/tmp/local-root"
    with pytest.raises(ST.StoresError, match=r"names no \[\[clusters\]\] entry 'label_one' \(declared: none\)"):
        cfg.destination("label_one")


def test_the_join_reads_the_root_it_is_given_else_registry_root(tmp_path, monkeypatch):
    a = tmp_path / "a"
    a.mkdir()
    root_a = _root(a, '[[clusters]]\nname = "label_one"\nhost = "login-a"\nshared = "/shared/a"\n')
    b = tmp_path / "b"
    b.mkdir()
    root_b = _root(b, '[[clusters]]\nname = "label_one"\nhost = "login-b"\nshared = "/shared/b"\n')
    y = _yaml(tmp_path)
    monkeypatch.setenv("REGISTRY_ROOT", root_b)
    assert ST.load_config(y, root=root_a).destination("label_one").host == "login-a", "an explicit root wins"
    assert ST.load_config(y).destination("label_one").host == "login-b", "else $REGISTRY_ROOT"


def test_one_joined_destination_alone_is_the_default(tmp_path):
    root = _root(tmp_path, '[[clusters]]\nname = "label_one"\nhost = "login-one"\nshared = "/shared/one"\n')
    cfg = ST.load_config(_yaml(tmp_path, YAML.replace("  label_two:\n    stores_rel: registry/stores\n    verify_cpus: 4\n  local_dest:\n"
                                                      "    shared: /tmp/local-root\n", "")), root=root)
    assert cfg.declared() == ["label_one"] and cfg.destination(None).host == "login-one"


def test_a_refused_registry_configuration_is_a_stores_refusal_naming_its_file(tmp_path):
    root = _root(tmp_path, '[[clusters]]\nname = "label_one"\nhost = "login-one"\nshared = "/shared/one"\n\n[unterminated\n')
    cfg = ST.load_config(_yaml(tmp_path), root=root)
    with pytest.raises(ST.StoresError, match=r"config\.toml: the registry configuration names the clusters a destination joins, and it is refused"):
        cfg.destination("label_one")


def test_a_destination_spelling_one_of_host_or_shared_keeps_the_old_rule(tmp_path):
    with pytest.raises(ST.StoresError, match=r"destination 'half' lacks `shared`"):
        ST.load_config(_yaml(tmp_path, YAML.replace("  label_one: {}\n", "  half:\n    host: login-half\n")))
    with pytest.raises(ST.StoresError, match=r"destination 'label_one' has unknown keys \['tint'\]"):
        ST.load_config(_yaml(tmp_path, YAML.replace("  label_one: {}\n", "  label_one:\n    tint: blue\n")))


def test_a_destination_without_repo_rel_names_the_clone_after_its_source_checkout(tmp_path):
    """The default mirror is the clone the repo-mirror tool fills: <shared>/repos/<the source checkout's directory name>, a linked worktree
    naming the checkout it was added from."""
    assert ST.Destination(name="x", host=None, shared="/tmp/root", source_repo="/checkouts/foo").repo_rel == "repos/foo"
    assert ST.Destination(name="x", host=None, shared="/tmp/root", source_repo="/checkouts/foo", repo_rel="repos/other").repo_rel == "repos/other"
    main = tmp_path / "foo"
    (main / ".git" / "worktrees" / "lane").mkdir(parents=True)
    wt = main / "worktrees" / "lane"
    wt.mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/lane\n")
    assert ST.checkout_dir_name(str(wt)) == "foo" and ST.checkout_dir_name(str(main)) == "foo"
    assert ST.Destination(name="x", host=None, shared="/tmp/root", source_repo=str(wt)).mirror_dir == "/tmp/root/repos/foo"
    assert ST.Destination(name="x", host=None, shared="/tmp/root").repo_rel == f"repos/{ST.checkout_dir_name(ST.REPO_ROOT)}"


def test_the_stores_command_joins_the_root_it_was_given(tmp_path, monkeypatch, capsys):
    """`registry --root R stores ... --dest L` joins R's [[clusters]] entry, whatever $REGISTRY_ROOT says, and refuses a label R names no
    entry for by name (exit 2)."""
    from sisteer.registry import cli
    root = _root(tmp_path, '[[clusters]]\nname = "label_one"\nhost = "login-one"\nshared = "/shared/one"\n')
    monkeypatch.setenv(ST.CONFIG_ENV, _yaml(tmp_path))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "config.toml").write_text('[[clusters]]\nname = "label_two"\nhost = "login-x"\nshared = "/shared/x"\n')
    monkeypatch.setenv("REGISTRY_ROOT", str(elsewhere))
    assert cli.main(["--root", root, "stores", "status", "--dest", "label_two", "--json"]) == 2
    err = capsys.readouterr().err
    assert "names no [[clusters]] entry 'label_two' (declared: label_one)" in err and root in err, err
