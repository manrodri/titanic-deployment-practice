#!/bin/bash

# Building packages and uploading them to a Gemfury repository

GEMFURY_URL=$GEMFURY_PUSH_URL

set -e

DIRS="$@"
BASE_DIR=$(pwd)

warn() {
    echo "$@" 1>&2
}

die() {
    warn "$@"
    exit 1
}

build() {

    for X in ./dist/*

    do
      [[ $X == *.tar.gz ]] &&  curl -F package=@"dist/$X" "$GEMFURY_URL" || die "Uploading package $PACKAGE_NAME failed"
      [[ $X == *.whl ]] && curl -F package=@"dist/$X" "$GEMFURY_URL" || die "Uploading package $PACKAGE_NAME failed"
    done
}

build