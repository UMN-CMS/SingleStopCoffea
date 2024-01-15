ENVNAME=coffeaenv
LCG_VIEW=LCG_104cuda
LCG_ARCH=x86_64-centos7-gcc11-opt
LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/${LCG_VIEW}/${LCG_ARCH}/setup.sh"

printf "Sourcing %s for achitecture %s\n" "${LCG_VIEW}"  "${LCG_ARCH}"
source $LCG_SETUP
printf "Python version is '%s'\n" "$(python3 --version)"
if [[ ! -d $ENVNAME ]]; then
    printf "Virtual environment does not exist, creating virtual environment\n"
    python3 -m venv --copies "$ENVNAME"
    printf "Created virtual environment %s\n" $ENVNAME
fi

source "$ENVNAME"/bin/activate
LOCALPATH=$ENVNAME$(python3 -c 'import sys; print(f"/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")')
export PYTHONPATH=${LOCALPATH}:$PYTHONPATH

if ! compgen -G "${ENVNAME}/lib/python*/*/*coffea*" > /dev/null; then
    printf "Upgrading installation tools\n"
    python3 -m pip install setuptools pip wheel --upgrade
    printf "Installing project\n"
    python3 -m pip install . --upgrade
    sed -i '1s/#!.*python$/#!\/usr\/bin\/env python3/' $NAME/bin/*
    sed -i '40s/.*/VIRTUAL_ENV="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}" )")" \&\& pwd)"/' $ENVNAME/bin/activate
    #sed -i "2a source ${LCG_SETUP}" $ENVNAME/bin/activate
    #sed -i "3a export PYTHONPATH=${LOCALPATH}:\$PYTHONPATH" $ENVNAME/bin/activate
fi

