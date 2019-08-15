#!/bin/bash

REL_SCRIPT_DIR=$(dirname "$0")
INITIAL_DIR=$(pwd)
cd $REL_SCRIPT_DIR
ABS_SCRIPT_DIR=$(pwd)

if [ $# -lt 1 ]; then
    echo 1>&2 "Usage: $0 <exercise_num>"
    echo 1>&2 "e.g. $0 1"
    exit 1
fi

# Start agent and add key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/i2dl_ss19

# Hard pull
git fetch origin
git reset --hard origin/master
git pull origin master

# Get exercise of choice
git submodule update --init -- exercise_$1


cd $INITIAL_DIR

