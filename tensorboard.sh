#!/bin/bash

if python -c "import tensorboard" > /dev/null
then
    python -m tensorboard.main --logdir=$1 &
    if which xdg-open > /dev/null
    then
	xdg-open http://localhost:6006
    else
	echo "Please install xdg-open."
    fi
else
    echo "Please install tensorboard."
fi
