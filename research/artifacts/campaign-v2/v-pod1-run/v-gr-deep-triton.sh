#!/usr/bin/env bash
# V-GR-DEEP's trimmed end (controller ruling 2026-10-10): finish the SilverTorch-triton arm only (resume), at campaign-v2.3;
# no V3, no official (C2 / C7 decided elsewhere).
LEG=v-gr-deep
TAG=campaign-v2.3
. "$(dirname "$(readlink -f "$0")")/common.sh"
stream goodreads deep "silvertorch|triton"
finish
