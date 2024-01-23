ENVNAME=coffeaenv
LCG_VIEW=LCG_104cuda
LCG_ARCH=x86_64-centos7-gcc11-opt
LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/${LCG_VIEW}/${LCG_ARCH}/setup.sh"
CONTAINER=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/cc7:x86_64

export CONDOR_CONFIG=".condor_config"


function source_lcg(){
    printf "Sourcing %s for achitecture %s\n" "${LCG_VIEW}"  "${LCG_ARCH}"
    source "$LCG_SETUP"
    printf "Python version is '%s'\n" "$(python3 --version)"
}


function activate_venv(){
    source "$ENVNAME"/bin/activate
    LOCALPATH=$ENVNAME$(python3 -c 'import sys; print(f"/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")')
    export PYTHONPATH=${LOCALPATH}:$PYTHONPATH
}

function create_venv(){
    python3 -m venv --copies "$ENVNAME"
    printf "Created virtual environment %s\n" $ENVNAME
    printf "Upgrading installation tools\n"
    python3 -m pip install setuptools pip wheel --upgrade
    printf "Installing project\n"
    python3 -m pip install . --upgrade
    sed -i '1s/#!.*python$/#!\/usr\/bin\/env python3/' $NAME/bin/*
    sed -i '40s/.*/VIRTUAL_ENV="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}" )")" \&\& pwd)"/' $ENVNAME/bin/activate
    #sed -i "2a source ${LCG_SETUP}" $ENVNAME/bin/activate
    #sed -i "3a export PYTHONPATH=${LOCALPATH}:\$PYTHONPATH" $ENVNAME/bin/activate
}

function create_venv_if_needed(){
    if [[ ! -d $ENVNAME ]]; then
        printf "Virtual environment does not exist, creating virtual environment\n"
        create_venv
    fi
}


function startup_with_lcg(){
    source_lcg
    create_venv_if_needed
    activate_venv
}

function main(){
    startup_with_lcg
}

main
