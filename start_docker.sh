#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  MultiSpec Docker — Script de arranque para Ubuntu 24.04
#  Instala dependencias del host, prepara cámaras y lanza container
#
#  Uso: chmod +x start_docker.sh && ./start_docker.sh
# ─────────────────────────────────────────────────────────────

set -e

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║    MultiSpec Docker — Iniciando...       ║"
echo "╚══════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Detección de GPU NVIDIA ────────────────────────────────────
echo "Detectando GPU NVIDIA..."
HAS_GPU=false
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    HAS_GPU=true
    echo "      ✓ GPU detectada: $GPU_NAME"
elif lspci 2>/dev/null | grep -qi "nvidia"; then
    # nvidia-smi no está pero hay GPU NVIDIA en el sistema
    HAS_GPU=true
    echo "      ✓ GPU NVIDIA detectada (drivers no instalados aún)"
else
    echo "      ℹ Sin GPU NVIDIA — modo CPU (super-resolución usará CPU)"
fi
echo ""

# Exporta para que docker-compose lo lea
export HAS_GPU

# ── 1. Dependencias del host ───────────────────────────────────
echo "[1/6] Comprobando dependencias del host..."

# Docker
if ! command -v docker &>/dev/null; then
    echo "      Instalando Docker..."
    sudo apt-get update -qq
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo usermod -aG docker "$USER"
    echo "      ✓ Docker instalado — puede que necesites cerrar sesión y volver a entrar"
else
    echo "      ✓ Docker $(docker --version | awk '{print $3}' | tr -d ',')"
fi

# NVIDIA Container Toolkit (solo si hay GPU NVIDIA)
if [ "$HAS_GPU" = "true" ]; then
    if ! dpkg -l | grep -q nvidia-container-toolkit 2>/dev/null; then
        echo "      Instalando NVIDIA Container Toolkit..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update -qq
        sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        echo "      ✓ NVIDIA Container Toolkit instalado"
    else
        echo "      ✓ NVIDIA Container Toolkit ya instalado"
    fi
else
    echo "      ℹ Sin GPU — omitiendo NVIDIA Container Toolkit"
fi

# Herramientas del host para cámaras
HOST_PKGS=()
for pkg in gphoto2 ffmpeg v4l-utils v4l2loopback-dkms v4l2loopback-utils; do
    if ! dpkg -l "$pkg" &>/dev/null; then
        HOST_PKGS+=("$pkg")
    fi
done

if [ ${#HOST_PKGS[@]} -gt 0 ]; then
    echo "      Instalando: ${HOST_PKGS[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y "${HOST_PKGS[@]}"
    echo "      ✓ Herramientas de cámara instaladas"
else
    echo "      ✓ Herramientas de cámara ya instaladas"
fi

# Headers del kernel (necesarios para v4l2loopback-dkms)
KERNEL_VER=$(uname -r)
HEADERS_PKG="linux-headers-${KERNEL_VER}"
if ! dpkg -l "$HEADERS_PKG" &>/dev/null; then
    echo "      Instalando headers del kernel ${KERNEL_VER}..."
    sudo apt-get install -y "$HEADERS_PKG"
    echo "      ✓ Headers instalados"
else
    echo "      ✓ Headers del kernel ya instalados"
fi

echo "      ✓ Todas las dependencias del host listas"

# ── 2. v4l2loopback en el HOST ────────────────────────────────
echo "[2/6] Cargando v4l2loopback en el host..."
if ! lsmod | grep -q v4l2loopback; then
    sudo modprobe v4l2loopback devices=1 video_nr=2 card_label="Canon UV" exclusive_caps=1
    echo "      ✓ /dev/video2 creado"
else
    echo "      ✓ v4l2loopback ya cargado"
fi

# Hacer v4l2loopback persistente entre reinicios
if [ ! -f /etc/modules-load.d/v4l2loopback.conf ]; then
    echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf > /dev/null
    echo 'options v4l2loopback devices=1 video_nr=2 card_label="Canon UV" exclusive_caps=1' | \
        sudo tee /etc/modprobe.d/v4l2loopback.conf > /dev/null
    echo "      ✓ v4l2loopback configurado para arrancar automáticamente"
fi

# ── 3. Liberar Canon de gvfs ──────────────────────────────────
echo "[3/6] Liberando Canon de gvfs..."
GVFS_PID=$(pgrep -f gvfsd-gphoto2 2>/dev/null || true)
if [ -n "$GVFS_PID" ]; then
    kill -9 $GVFS_PID 2>/dev/null || true
    sleep 1
    echo "      ✓ gvfsd-gphoto2 detenido (PID $GVFS_PID)"
else
    echo "      ✓ Sin bloqueo gvfs"
fi

# ── 4. Stream Canon → /dev/video2 ────────────────────────────
echo "[4/6] Arrancando stream Canon UV..."
GPHOTO_PID=""
if lsusb | grep -qi canon; then
    gphoto2 --stdout --capture-movie 2>/dev/null | \
        ffmpeg -i - -vcodec rawvideo -pix_fmt yuyv422 -threads 0 -f v4l2 /dev/video2 \
        -loglevel quiet &
    GPHOTO_PID=$!
    echo "      ✓ Canon streaming (PID $GPHOTO_PID)"
    sleep 2
else
    echo "      ⚠ Canon no detectada — continuando sin ella"
fi

# ── 5. Detectar cámaras ───────────────────────────────────────
echo "[5/6] Detectando cámaras..."

get_device_name() {
    v4l2-ctl --device="$1" --info 2>/dev/null | grep "Card type" | sed 's/.*: //' | xargs
}
find_device_by_name() {
    for dev in /dev/video*; do
        [ -e "$dev" ] || continue
        name=$(get_device_name "$dev")
        if echo "$name" | grep -qiE "$1"; then echo "$dev"; return; fi
    done
    echo ""
}

DEV_UV=2
echo "      UV      → /dev/video$DEV_UV (Canon via gphoto2)"

DEV_THERMAL_PATH=$(find_device_by_name "TOPDON|TC001|topdon")
[ -z "$DEV_THERMAL_PATH" ] && DEV_THERMAL_PATH=$(find_device_by_name "UVC|USB Camera|Infrared")
DEV_THERMAL=$(echo "$DEV_THERMAL_PATH" | grep -o '[0-9]*$'); DEV_THERMAL=${DEV_THERMAL:-0}
echo "      Térmica → /dev/video$DEV_THERMAL ($DEV_THERMAL_PATH)"

# Visible: primera webcam que no sea UV ni térmica
DEV_VISIBLE_PATH=""
for dev in /dev/video*; do
    [ -e "$dev" ] || continue
    idx=$(echo "$dev" | grep -o '[0-9]*$')
    if [ "$idx" != "$DEV_UV" ] && [ "$idx" != "$DEV_THERMAL" ]; then
        name=$(get_device_name "$dev")
        [ -n "$name" ] && { DEV_VISIBLE_PATH="$dev"; break; }
    fi
done
DEV_VISIBLE=$(echo "$DEV_VISIBLE_PATH" | grep -o '[0-9]*$'); DEV_VISIBLE=${DEV_VISIBLE:-4}
echo "      Visible → /dev/video$DEV_VISIBLE ($DEV_VISIBLE_PATH)"

cat > "$SCRIPT_DIR/camera_config.json" << JSONEOF
{
  "uv":      $DEV_UV,
  "thermal": $DEV_THERMAL,
  "visible": $DEV_VISIBLE
}
JSONEOF
echo "      ✓ camera_config.json generado"

# Crea state.json vacío si no existe
[ -f "$SCRIPT_DIR/state.json" ] || echo "{}" > "$SCRIPT_DIR/state.json"

# ── 6. Lanza el container ─────────────────────────────────────
echo ""
echo "[6/6] Lanzando container Docker..."
echo "──────────────────────────────────────────"

# Elige el compose file según si hay GPU
if [ "$HAS_GPU" = "true" ]; then
    COMPOSE_FILE="docker-compose.yml"
    echo "      Modo: GPU (CUDA)"
else
    COMPOSE_FILE="docker-compose.cpu.yml"
    echo "      Modo: CPU (sin CUDA)"
fi

# Comprueba que el usuario está en el grupo docker
DOCKER_CMD="docker compose -f $COMPOSE_FILE"
if ! groups "$USER" | grep -q docker; then
    echo "⚠ El usuario $USER no está en el grupo docker."
    echo "  Ejecuta: newgrp docker"
    echo "  O cierra sesión y vuelve a entrar."
    echo "  Lanzando con sudo como fallback..."
    DOCKER_CMD="sudo docker compose -f $COMPOSE_FILE"
fi

cd "$SCRIPT_DIR"
$DOCKER_CMD up --build

# Limpieza al salir (Ctrl+C)
cleanup() {
    echo ""
    echo "Deteniendo..."
    [ -n "$GPHOTO_PID" ] && kill $GPHOTO_PID 2>/dev/null || true
    cd "$SCRIPT_DIR" && $DOCKER_CMD down 2>/dev/null || true
}
trap cleanup EXIT
