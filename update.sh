#!/bin/bash

# change to directory of this script
cd "$(dirname "$0")"

# update from github
git pull

# generate files
./build.sh
