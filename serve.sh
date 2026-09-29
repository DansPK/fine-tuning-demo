#!/usr/bin/env bash
#
# Build the fine-tuned model in Ollama and expose it through a Cloudflare quick
# tunnel (no Cloudflare account required).
#
# Ollama, cloudflared and zstd are installed automatically if they are missing.
#
# Usage:
#   ./serve.sh
#
# Env overrides:
#   MODEL=qwen3-json-extractor   Ollama model name
#   PORT=11434                   Ollama port
#
# WARNING: the tunnel has NO authentication. Anyone with the URL can query the
# model. Stop the script with Ctrl+C when you are done.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-qwen3-json-extractor}"
PORT="${PORT:-11434}"
export OLLAMA_HOST="127.0.0.1:${PORT}"
OLLAMA_URL="http://${OLLAMA_HOST}"

OS="$(uname -s)"

install_zstd() {
    echo "zstd not found - installing..."
    if [ "$OS" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
        brew install zstd
        return
    fi

    local sudo=""
    if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
        sudo="sudo"
    fi

    if command -v apt-get >/dev/null 2>&1; then
        $sudo apt-get update -y && $sudo apt-get install -y zstd
    elif command -v dnf >/dev/null 2>&1; then
        $sudo dnf install -y zstd
    elif command -v yum >/dev/null 2>&1; then
        $sudo yum install -y zstd
    elif command -v pacman >/dev/null 2>&1; then
        $sudo pacman -Sy --noconfirm zstd
    elif command -v zypper >/dev/null 2>&1; then
        $sudo zypper install -y zstd
    elif command -v apk >/dev/null 2>&1; then
        $sudo apk add zstd
    else
        echo "warning: could not auto-install zstd; continuing anyway."
    fi
}

install_ollama() {
    echo "Ollama not found - installing..."
    if [ "$OS" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
        brew install ollama
    elif [ "$OS" = "Linux" ]; then
        # Official installer: sets up the binary and a systemd service.
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "error: don't know how to auto-install Ollama on $OS."
        echo "  Install it from https://ollama.com/download and re-run."
        exit 1
    fi
}

install_cloudflared() {
    echo "cloudflared not found - installing..."
    if [ "$OS" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
        brew install cloudflared
        return
    fi

    local arch
    case "$(uname -m)" in
        x86_64 | amd64) arch=amd64 ;;
        aarch64 | arm64) arch=arm64 ;;
        *)
            echo "error: unsupported architecture $(uname -m)."
            exit 1
            ;;
    esac

    local tmp
    tmp="$(mktemp)"
    curl -fsSL \
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${arch}" \
        -o "$tmp"

    if [ -w /usr/local/bin ]; then
        install -m 0755 "$tmp" /usr/local/bin/cloudflared
    elif command -v sudo >/dev/null 2>&1; then
        sudo install -m 0755 "$tmp" /usr/local/bin/cloudflared
    else
        mkdir -p "$HOME/.local/bin"
        install -m 0755 "$tmp" "$HOME/.local/bin/cloudflared"
        export PATH="$HOME/.local/bin:$PATH"
        echo "Installed to ~/.local/bin (added to PATH for this run)."
    fi
    rm -f "$tmp"
}

command -v zstd >/dev/null 2>&1 || install_zstd
command -v ollama >/dev/null 2>&1 || install_ollama
command -v cloudflared >/dev/null 2>&1 || install_cloudflared

# The Modelfile's FROM line must point at the GGUF exported by fine-tune.ipynb.
GGUF="$(awk 'tolower($1) == "from" { print $2 }' Modelfile)"
if [ ! -f "$GGUF" ]; then
    echo "error: Modelfile expects a GGUF at '$GGUF' but it was not found."
    echo "  Copy the .gguf exported by fine-tune.ipynb into this folder."
    exit 1
fi

# Start the Ollama server if it is not already running.
if ! ollama list >/dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve >/tmp/ollama.log 2>&1 &
    for _ in $(seq 1 20); do
        ollama list >/dev/null 2>&1 && break
        sleep 0.5
    done
fi

# Create (or update) the model from the Modelfile.
echo "Creating Ollama model '${MODEL}'..."
ollama create "$MODEL" -f Modelfile

echo
echo "Ollama is serving the model on ${OLLAMA_URL}"
echo "Opening a Cloudflare quick tunnel (the public URL appears below)..."
echo "  WARNING: no authentication - anyone with the URL can query the model."
echo

# --http-host-header rewrites the Host seen by Ollama to a localhost value.
# Without it Ollama rejects tunnelled requests with an empty 403 (its
# DNS-rebinding protection only trusts localhost Host headers).
exec cloudflared tunnel --url "${OLLAMA_URL}" --http-host-header "localhost:${PORT}"
