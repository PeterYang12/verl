#!/usr/bin/env bash
# Start Ray inside the container. Run once per node, head first.
#   bash examples/grpo_trainer/start_ray_rocm.sh head      # on the head node
#   bash examples/grpo_trainer/start_ray_rocm.sh worker    # on every other node
# The node address is the one on the 10.48.0.0/16 network (interface spur0).
# Megatron ranks are assigned in the order of these addresses, lowest first.
set -euo pipefail
ROLE=${1:?usage: start_ray_rocm.sh head|worker}
HEAD_IP=${HEAD_IP:-10.48.0.3}
PORT=${PORT:-6380}
# The image has no `ip` command, so ask the kernel through Python.
IP=$(python3 -c "import socket,fcntl,struct,sys; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); print(socket.inet_ntoa(fcntl.ioctl(s.fileno(),0x8915,struct.pack('256s',sys.argv[1][:15].encode()))[20:24]))" "${RAY_IFNAME:-spur0}" 2>/dev/null || true)
[ -n "$IP" ] || { echo "no address on ${RAY_IFNAME:-spur0}"; exit 1; }
ray stop >/dev/null 2>&1 || true
case "$ROLE" in
    head)   [ "$IP" = "$HEAD_IP" ] || { echo "this node is $IP, head is expected at $HEAD_IP (set HEAD_IP to change)"; exit 1; }
            ray start --head --node-ip-address="$IP" --port="$PORT" --num-gpus=8 ;;
    worker) ray start --address="${HEAD_IP}:${PORT}" --node-ip-address="$IP" --num-gpus=8 ;;
    *)      echo "usage: start_ray_rocm.sh head|worker"; exit 1 ;;
esac
