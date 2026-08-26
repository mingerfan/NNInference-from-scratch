#!/bin/sh

set -eu
export LC_ALL=C

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    printf '%s\n' "Usage: validate-commit-message.sh <main|branch> <subject> [context]" >&2
    exit 1
fi

mode=$1
subject=$2
context=${3:-$mode commit}

if [ "${#subject}" -gt 72 ]; then
    printf '%s\n' \
        "Commit message rejected for $context: subject exceeds 72 characters." \
        "Received (${#subject} characters): $subject" >&2
    exit 1
fi

case "$mode" in
    main)
        pattern='^(Chapter[0-9]+\.[0-9]+\.[0-9]+|(Docs|Fix|Refactor|Test|Bench|Perf|Build|CI|Chore|Revert)): [A-Z].+$'
        expected='Chapter1.1.1: Subject or Docs: Subject'
        ;;
    branch)
        pattern='^(Add|Fix|Test|Refactor|Docs|Bench|Perf|Build|CI|Chore|Spike|Wip)\([a-z0-9][a-z0-9-]*\): [A-Z].+$'
        expected='Add(scope): Subject or Fix(scope): Subject'
        ;;
    *)
        printf '%s\n' "Unknown commit message mode: $mode" >&2
        exit 1
        ;;
esac

if ! printf '%s\n' "$subject" | grep -Eq "$pattern"; then
    printf '%s\n' \
        "Commit message rejected for $context." \
        "Received: $subject" \
        "Expected: $expected" \
        "See COMMIT_CONVENTION.md for all allowed prefixes." >&2
    exit 1
fi
