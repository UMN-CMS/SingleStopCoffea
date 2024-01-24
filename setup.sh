#!/usr/bin/bash
ENVNAME=coffeaenv
LCG_VIEW=LCG_104cuda
LCG_ARCH=x86_64-centos7-gcc11-opt
LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/${LCG_VIEW}/${LCG_ARCH}/setup.sh"
CONTAINER=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/cc7:x86_64

export CONDOR_CONFIG=".condor_config"


function create_venv(){
    python3 -m venv --copies "$ENVNAME"
    printf "Created virtual environment %s\n" $ENVNAME
    printf "Upgrading installation tools\n"
    python3 -m pip install setuptools pip wheel --upgrade
    printf "Installing project\n"
    python3 -m pip install . --upgrade
    sed -i '1s/#!.*python$/#!\/usr\/bin\/env python3/' $NAME/bin/*
    sed -i '40s/.*/VIRTUAL_ENV="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}" )")" \&\& pwd)"/' $ENVNAME/bin/activate
}

function activate_venv(){
    source "$ENVNAME"/bin/activate
    local localpath=$ENVNAME$(python3 -c 'import sys; print(f"/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")')
    export PYTHONPATH=${localpath}:$PYTHONPATH
}

function source_lcg(){
    printf "Sourcing %s for achitecture %s\n" "${LCG_VIEW}"  "${LCG_ARCH}"
    source "$LCG_SETUP"
    printf "Python version is '%s'\n" "$(python3 --version)"
}

function startup_with_lcg(){
    source_lcg
    if [[ ! -d $ENVNAME ]]; then
        printf "Virtual environment does not exist, creating virtual environment\n"
        create_venv
    fi
    activate_venv
}

function startup_with_container(){
    local in_apptainer=${APPTAINER_COMMAND:-false}
    if [ "$in_apptainer"  = false ]; then
        printf "Not in apptainer, running command to start container %s.\n" "${CONTAINER}";
        apptainer exec --pid --ipc --contain --bind /cvmfs --bind /etc/hosts --home .:/srv --pwd /srv "$CONTAINER" /bin/bash --rcfile <(echo 'source setup.sh container')
    else
        printf "Currently in apptainer\n"
        if [[ ! -d $ENVNAME ]]; then
            printf "Virtual environment does not exist, creating virtual environment\n"
            create_venv
        fi
        activate_venv
    fi
}

function main(){
    local mode=${1:-lcg}
    case "$mode" in
        container )
            startup_with_container
            ;;
        lcg )
            startup_with_lcg
            ;;
        * )
            printf "Unknown mode\n"
            return 1
            ;;
    esac
}

main "$@"
