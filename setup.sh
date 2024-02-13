LCG_VIEW=LCG_103cuda
LCG_ARCH=x86_64-centos7-gcc11-opt

source "/cvmfs/sft.cern.ch/lcg/views/${LCG_VIEW}/${LCG_ARCH}/setup.sh"
if [[ ! -d env ]]; then
   printf "Creating virtual environment\n"
   python3 -m venv env
fi
source env/bin/activate
if ! compgen -G "env/lib/python*/*/*coffea*" > /dev/null; then
    python3 -m pip install .
fi
export PYTHONPATH=$(realpath env/lib/python3.9/site-packages/):$PYTHONPATH


#DIR=$(pwd)
#if [[ ! -d $CMSSW_REL ]]; then 
#    printf "Creating CMSSW area for appropriate python version\n"
#    scramv1 project CMSSW $CMSSW_REL
#fi
#cd $CMSSW_REL
#eval "$(scramv1 runtime -sh)"
#cd $DIR
#if [[ ! -d env ]]; then
#   printf "Creating virtual environment\n"
#   python3 -m venv env
#fi
#source env/bin/activate
#if ! compgen -G "env/lib/python*/*/*coffea*" > /dev/null; then
#    python3 -m pip install .
#fi




