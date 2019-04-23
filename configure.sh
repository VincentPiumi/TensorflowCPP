#!/bin/bash

CURR_DIR=`pwd`
DEPENDENCIES="curl cmake git unzip autoconf autogen automake libtool mlocate zlib1g-dev g++-7 python python3-numpy python3-dev python3-pip python3-wheel wget"
BAZEL_SRC="bazel-0.21.0-installer-linux-x86_64.sh"
CMAKE_TAR="cmake-3.13.4.tar.gz"
CMAKE_DIR="cmake-3.13.4"
RT=0

run_return_check()
{
    if [ $RT -eq 1 ]; then
	echo "$1 $2 command FAILED"
	exit 1
    fi
}

run_dpkg_check()
{
    if dpkg -s $1 > /dev/null 2>&1 ; then
	echo "[install] $1 found"
    else
	echo "[install] $1"
	sudo apt -f install $1
	RT=$?
	run_return_check $1 install
    fi
}

run_bazel_install()
{
    wget https://github.com/bazelbuild/bazel/releases/download/0.21.0/bazel-0.21.0-installer-linux-x86_64.sh
    if [ $? -eq 1 ]; then
	RT=1
    fi
    run_return_check BAZEL download
    chmod +x $BAZEL_SRC
    if [ $? -eq 1 ]; then
	RT=1
    fi
    run_return_check BAZEL executable
    
    if [ -f "$BAZEL_SRC" ]; then 
	echo "[install] BAZEL"
	bash $BAZEL_SRC --user
	export PATH="$PATH:$HOME/bin"
    else
	RT=1
    fi
}

run_cmake_install()
{
    wget https://github.com/Kitware/CMake/releases/download/v3.13.4/cmake-3.13.4.tar.gz
    if [ $? -eq 1 ]; then
	RT=1
    fi
    run_return_check CMAKE download
    tar xzvf $CMAKE_TAR
    if [ $? -eq 1 ]; then
	RT=1
    fi
    run_return_check CMAKE extraction

    if [ -d "$CMAKE_DIR" ]; then
	echo "[install] CMAKE"
	cd $CMAKE_DIR
	./bootstrap
	run_return_check CMAKE bootstrap
	make
	run_return_check CMAKE make
	sudo make install
	run_return_check CMAKE make_install
	cd ..
    else
	RT=1
    fi
}

run_tensorflow_install()
{
    echo "[install] TENSORFLOW (this may take a while..)"
    
    git clone https://github.com/FloopCZ/tensorflow_cc.git
    if [ $? -eq 1 ]; then
	RT=1
    fi
    run_return_check TENSORFLOW git_clone

    cd tensorflow_cc/tensorflow_cc/
    if [ $? -eq 1 ]; then
	RT=1
    fi
    run_return_check TENSORFLOW git_clone

    mkdir build && cd build
    if [ $? -eq 1 ]; then
	RT=1
    fi
    run_return_check TENSORFLOW git_clone

    cmake -DTENSORFLOW_STATIC=OFF -DTENSORFLOW_SHARED=ON .. 
    if [ $? -eq 1 ]; then
	RT=1
    fi
    run_return_check TENSORFLOW git_clone

    make && sudo make install
    if [ $? -eq 1 ]; then
	RT=1
    fi
    run_return_check TENSORFLOW git_clone

    cd ../../..
}

run_dependencies()
{
    for i in $DEPENDENCIES; do
	run_dpkg_check $i
    done;
    sudo updatedb
    RT=$?

}

run_bazel_check()
{
    BAZEL_VERSION="$(bazel version | grep "0.21.0")"
    if [ $? -eq 1 ]; then
	run_bazel_install
	run_return_check BAZEL install
    fi
}

run_cmake_check()
{
    CMAKE_VERSION="$(cmake --version | grep "3.13.4")"
    if [ $? -eq 1 ]; then
	run_cmake_install
	run_return_check CMAKE install
    fi
}

run_main()
{
    # run_dependencies
    # run_return_check DEPENDENCIES
    run_bazel_check
    run_cmake_check
    # run_tensorflow_install
    # run_return_check TENSORFLOW install    
}

run_main
