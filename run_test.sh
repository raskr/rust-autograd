#!/usr/bin/env bash

# Don't use "cargo test -all"
for filename in `find ./tests -name "test_*.rs" -type f`
do
    cargo test --test ${filename:8:${#filename}-11}
done