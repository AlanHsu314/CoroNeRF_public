######################################################
### SCRIPTS TO RESOLVE CONFIG PATHS
######################################################

from __future__ import annotations
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import os
import re

_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")
_WINDOWS_MISSING_COLON_RE = re.compile(r"^[A-Za-z][\\/]")

@dataclass(frozen=True)
class ResolvePathsSpec:
    """
    Controls what gets treated as a path and how it's resolved.
    """
    # Keys that are treated as path-like if present anywhere in nested dicts.
    # IMPORTANT:
    # All input/resource paths must be explicitly listed here.
    # Do NOT rely on suffix-based matching like "_dir".
    path_keys: tuple[str, ...] = (
        "base_dir",
        "dataset_dir",
        "dataset_base_dir",
        "LUT_dir",
        "resume_run_dir",
    )

    # Additionally treat any key that *endswith* one of these suffixes as a path.
    path_key_suffixes: tuple[str, ...] = ("_path", "_file")

    # If True, attempt to resolve even if the path doesn't exist on disk.
    allow_nonexistent: bool = True

    # If True, convert Path objects back to strings (nice for YAML/JSON and DotDict usage)
    return_str: bool = True


def _is_path_key(key: str, spec: ResolvePathsSpec) -> bool:
    if key in spec.path_keys:
        return True
    return any(key.endswith(suf) for suf in spec.path_key_suffixes)


def _expand(p: str) -> str:
    # Expand "~" and "$ENV_VAR"
    return os.path.expandvars(os.path.expanduser(p))

def _normalize_portable_path_string(value: str) -> str:
    """
    Normalize path strings before pathlib sees them.

    Handles a common local YAML typo:
        d/Users2/... -> D:/Users2/...

    Also keeps Windows absolute paths portable when this code is inspected
    from POSIX-like environments.
    """
    s = _expand(str(value)).strip()

    # Convert backslashes for easier regex/prefix handling.
    s = s.replace("\\", "/")

    # Coerce "d/Users2/..." into "D:/Users2/...".
    # This prevents pathlib from treating it as a relative path.
    if _WINDOWS_MISSING_COLON_RE.match(s) and not _WINDOWS_ABS_RE.match(s):
        first = s[0]
        rest = s[2:]
        s = f"{first.upper()}:/{rest}"

    return s


def _is_portable_absolute_path(value: str) -> bool:
    s = _normalize_portable_path_string(value)
    return Path(s).is_absolute() or bool(_WINDOWS_ABS_RE.match(s))

def _resolve_one(value: Any, root: Path, spec: ResolvePathsSpec) -> Any:
    if value is None:
        return None

    if isinstance(value, Path):
        raw = str(value)
    elif isinstance(value, str):
        raw = value.strip()
        if raw == "":
            return value
        if "\n" in raw:
            return value
    else:
        return value

    s = _normalize_portable_path_string(raw)

    # If it is an absolute POSIX path or a Windows absolute path, do not join
    # against root. This is important for posthoc analysis on copied runs.
    if _is_portable_absolute_path(s):
        p = Path(s)
    else:
        p = root / s

    if spec.allow_nonexistent:
        try:
            p = p.resolve(strict=False)
        except Exception:
            p = Path(os.path.normpath(str(p)))
    else:
        p = p.resolve(strict=True)

    return str(p) if spec.return_str else p


def resolve_cfg_paths(cfg: Any, root: str | Path, spec: ResolvePathsSpec = ResolvePathsSpec()) -> Any:
    """
    Recursively resolve path-like fields inside a config object.

    - cfg can be a dict, DotDict, or any Mapping-like object.
    - root is usually the run_dir (directory containing config.yaml), not cwd.
    - This mutates cfg in-place when possible, and also returns it.
    """
    root_path = Path(root).expanduser()
    if not root_path.is_absolute():
        root_path = root_path.resolve()

    def walk(obj: Any, parent_key: str | None = None) -> Any:
        # Mapping (dict / DotDict)
        if isinstance(obj, Mapping):
            for k in list(obj.keys()):
                v = obj[k]
                if isinstance(k, str) and _is_path_key(k, spec):
                    obj[k] = _resolve_one(v, root_path, spec)
                else:
                    obj[k] = walk(v, parent_key=str(k) if isinstance(k, str) else None)
            return obj

        # List / tuple
        if isinstance(obj, list):
            for i in range(len(obj)):
                obj[i] = walk(obj[i], parent_key=parent_key)
            return obj

        if isinstance(obj, tuple):
            return tuple(walk(x, parent_key=parent_key) for x in obj)

        # Leaf value: only resolve if caller passed a key context suggesting path
        if parent_key is not None and _is_path_key(parent_key, spec):
            return _resolve_one(obj, root_path, spec)

        return obj

    return walk(cfg)


# helpers below are for remapping file paths when copying cluster to local benchmark runs

def _path_exists(value: Any) -> bool:
    if value is None:
        return False
    try:
        return Path(str(value)).expanduser().exists()
    except Exception:
        return False


def _split_any_path(value: Any) -> list[str]:
    """
    Split POSIX or Windows-looking paths into comparable components.

    This is intentionally separator-based so it can handle paths produced on
    Linux but loaded on Windows, or vice versa.
    """
    s = str(value).strip()
    if not s:
        return []

    s = os.path.expandvars(os.path.expanduser(s))
    s = s.replace("\\", "/")

    # Drop empty components caused by leading "/" or repeated separators.
    return [part for part in s.split("/") if part not in {"", "."}]


def _find_subsequence(parts: list[str], marker_parts: list[str]) -> int | None:
    if not marker_parts:
        return None

    parts_l = [p.lower() for p in parts]
    marker_l = [p.lower() for p in marker_parts]

    n = len(marker_l)
    for i in range(0, len(parts_l) - n + 1):
        if parts_l[i:i + n] == marker_l:
            return i

    return None


def _join_any(root: Any, suffix_parts: list[str]) -> Path:
    root_s = _normalize_portable_path_string(str(root))
    p = Path(root_s)
    for part in suffix_parts:
        p = p / part
    return p


def _remap_path_value(
    key: str,
    value: Any,
    path_remap: Mapping[str, Any],
    root: Path,
    spec: ResolvePathsSpec,
) -> Any:
    """
    Remap one path-like value using, in order:

    1. exact key overrides
    2. prefix rewrites
    3. suffix-marker remapping relative to a local root

    By default, remapping is only applied when the current path does not exist.
    """
    if value is None:
        return None

    if isinstance(value, Path):
        raw = str(value)
    elif isinstance(value, str):
        raw = value.strip()
        if raw == "" or "\n" in raw:
            return value
    else:
        return value

    enabled = bool(path_remap.get("enabled", True))
    if not enabled:
        return value

    only_if_missing = bool(path_remap.get("only_if_missing", True))
    if only_if_missing and _path_exists(raw):
        return value

    overrides = (
        path_remap.get("overrides", None)
        or path_remap.get("path_overrides", None)
        or {}
    )
    if key in overrides and overrides[key] is not None:
        return _resolve_one(overrides[key], root=root, spec=spec)

    # Prefix rewrites are useful when you know the old and new workspace roots.
    raw_norm = _normalize_portable_path_string(raw)
    raw_norm_lower = raw_norm.lower()

    for rule in path_remap.get("prefix_rewrites", []) or []:
        old = rule.get("from", None)
        new = rule.get("to", None)
        if old is None or new is None:
            continue

        old_norm = _normalize_portable_path_string(str(old)).rstrip("/")
        new_norm = _normalize_portable_path_string(str(new)).rstrip("/")

        if raw_norm_lower.startswith(old_norm.lower()):
            suffix = raw_norm[len(old_norm):].lstrip("/")
            rewritten = new_norm if not suffix else f"{new_norm}/{suffix}"
            return _resolve_one(rewritten, root=root, spec=spec)

    # Suffix-marker remapping is the most robust for cluster/local moves:
    # e.g. old/.../research/nerf/datasets/master -> local_root/datasets/master.
    local_root = (
        path_remap.get("local_root", None)
        or path_remap.get("repo_root", None)
        or path_remap.get("workspace_root", None)
    )

    if local_root is not None:
        local_root_s = str(local_root).replace("\\", "/")
        if _WINDOWS_MISSING_COLON_RE.match(local_root_s) and not _WINDOWS_ABS_RE.match(local_root_s):
            logging.getLogger("coroNeRF.resolve_paths").warning(
                "path_remap.local_root looked like a Windows drive path without a colon: %s. "
                "Interpreting it as %s",
                local_root,
                _normalize_portable_path_string(local_root_s),
            )

    suffix_markers = dict(path_remap.get("suffix_markers", {}) or {})
    marker = suffix_markers.get(key, None)

    if marker is None:
        default_markers = dict(path_remap.get("default_suffix_markers", {}) or {})
        marker = default_markers.get(key, None)

    if marker is None and key == "dataset_dir":
        marker = path_remap.get("dataset_marker", None)
    if marker is None and key == "LUT_dir":
        marker = path_remap.get("lut_marker", None)

    if local_root is not None and marker is not None:
        parts = _split_any_path(raw)
        marker_parts = _split_any_path(marker)
        idx = _find_subsequence(parts, marker_parts)

        if idx is not None:
            suffix_parts = parts[idx:]
            rewritten = _join_any(local_root, suffix_parts)
            return _resolve_one(rewritten, root=root, spec=spec)

    return value


def remap_cfg_paths(
    cfg: Any,
    path_remap: Mapping[str, Any] | None,
    root: str | Path,
    spec: ResolvePathsSpec = ResolvePathsSpec(),
) -> Any:
    """
    Recursively remap path-like fields in a loaded config.

    This is meant for posthoc analysis/figure generation when completed runs
    contain absolute paths from another machine. It mutates cfg in-place when
    possible and also returns it.
    """
    if not path_remap:
        return cfg

    root_path = Path(root).expanduser()
    if not root_path.is_absolute():
        root_path = root_path.resolve()

    def walk(obj: Any, parent_key: str | None = None) -> Any:
        if isinstance(obj, Mapping):
            for k in list(obj.keys()):
                v = obj[k]
                if isinstance(k, str) and _is_path_key(k, spec):
                    obj[k] = _remap_path_value(
                        key=str(k),
                        value=v,
                        path_remap=path_remap,
                        root=root_path,
                        spec=spec,
                    )
                else:
                    obj[k] = walk(v, parent_key=str(k) if isinstance(k, str) else None)
            return obj

        if isinstance(obj, list):
            for i in range(len(obj)):
                obj[i] = walk(obj[i], parent_key=parent_key)
            return obj

        if isinstance(obj, tuple):
            return tuple(walk(x, parent_key=parent_key) for x in obj)

        if parent_key is not None and _is_path_key(parent_key, spec):
            return _remap_path_value(
                key=str(parent_key),
                value=obj,
                path_remap=path_remap,
                root=root_path,
                spec=spec,
            )

        return obj

    return walk(cfg)



