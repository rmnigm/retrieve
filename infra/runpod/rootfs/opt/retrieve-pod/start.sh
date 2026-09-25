#!/usr/bin/env bash
set -uo pipefail

export -p | grep -vE '^declare -x (HOME|HOSTNAME|OLDPWD|PWD|SHLVL|TERM|_)=' > /etc/retrieve-pod.env
chmod 600 /etc/retrieve-pod.env

mkdir -p /workspace/.pod-home
printf '%s\n' "${PUBLIC_KEY:-}" "${EXTRA_PUBLIC_KEY:-}" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A >/dev/null
/usr/sbin/sshd

nohup herdr server >> /workspace/.pod-home/herdr.log 2>&1 &
/opt/retrieve-pod/bootstrap.sh >> /workspace/.pod-home/bootstrap.log 2>&1 &

exec sleep infinity
