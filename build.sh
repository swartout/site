#!/bin/bash

# --------------------------------- config -----------------------------------
# note - no trailing slashs for any of these

# out directory: where the generated files are stored
out="out"

# the directories to generate from
directories="root posts notes draft"

# directory for static assets
static="static"
# --------------------------------- config -----------------------------------

# pandoc convert func
convert () {
  pandoc -s --mathjax -f markdown -t html -B pandoc/body -H pandoc/header --template pandoc/template.html -o "$2" "$1"
}

# clear output directory
rm -rf $out

# copy assets
mkdir $out
cp -r $static "$out/$static"
cp favicon.ico "$out/favicon.ico"

# generate files
for DIR in $directories; do
    mkdir "$out/$DIR"
    DIR="$DIR/"
    for FILE in $(ls $DIR); do
        input="$DIR$FILE"
        output="$out/${DIR#root/}${FILE%.md}.html"
        convert $input $output
    done
done
