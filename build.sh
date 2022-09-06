#!/bin/sh

set -eu

CWD=$(basename "$PWD")

build() {
    docker build . --tag "$CWD"
}

clean() {
    docker system prune -f
}

dev() {
    mkdir -p output
    mkdir -p input
    docker run --rm --gpus=all --entrypoint=sh \
        -v huggingface:/home/huggingface/.cache/huggingface \
        -v "$PWD"/output:/home/huggingface/output \
        -v "$PWD"/input:/home/huggingface/input \
        -it "$CWD"
}

run() {
    shift
    mkdir -p output
    mkdir -p input
    docker run --rm --gpus=all \
        -v huggingface:/home/huggingface/.cache/huggingface \
        -v "$PWD"/output:/home/huggingface/output \
        -v "$PWD"/input:/home/huggingface/input \
        "$CWD" "$@"
}

case ${1:-build} in
    build) build ;;
    clean) clean ;;
    dev) dev "$@" ;;
    run) run "$@" ;;
    *) echo "$0: No command named '$1'" ;;
esac
