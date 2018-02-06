#!/bin/bash
TMPROOT=/tmp/poznej-fi-2018
PROJECT_ROOT=$TMPROOT/ai-workshop
mv $PROJECT_ROOT $PROJECT_ROOT.`date +'%s'`
mkdir -p $TMPROOT
git clone https://github.com/effa/ai-search-workshop $PROJECT_ROOT
~/.local/bin/jupyter notebook --no-browser
