# Keep every cache and temp file inside the project folder (for machines where only this folder is writable).
#   . scripts/local_env.sh      (from the project folder; scripts/gpu_queue.sh loads it automatically)
# Redirects HOME (pip, PyTorch, CUDA and other per-user caches) to .home/ and TMPDIR to .tmp/, and activates
# .venv/ if it exists.
_P=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export HOME="$_P/.home"
export XDG_CACHE_HOME="$HOME/.cache" XDG_CONFIG_HOME="$HOME/.config" XDG_DATA_HOME="$HOME/.local/share"
export PIP_CACHE_DIR="$HOME/.cache/pip" TORCH_HOME="$HOME/.cache/torch" CUDA_CACHE_PATH="$HOME/.nv/ComputeCache"
export MPLCONFIGDIR="$HOME/.config/matplotlib" PYTHONUSERBASE="${PYTHONUSERBASE:-$HOME/.local}"
export TMPDIR="$_P/.tmp" TMP="$_P/.tmp" TEMP="$_P/.tmp"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$TMPDIR"
[ -f "$_P/.venv/bin/activate" ] && [ -z "${VIRTUAL_ENV:-}" ] && . "$_P/.venv/bin/activate"
unset _P
