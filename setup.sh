#!/usr/bin/env bash

#ENVNAME=coffeaenv
declare -A env_configs
env_configs[coffea,venv]="coffeaenv"
env_configs[coffea,container]="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest"
env_configs[torch,venv]="cmsmlenv"
env_configs[torch,container]="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:3.10"


function activate_venv(){
    local env=$1
    source "$env"/bin/activate
    local localpath=$env$(python3 -c 'import sys; print(f"/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")')
    export PYTHONPATH=${localpath}:$PYTHONPATH
}

function create_venv(){
    local env=$1
    export TMPDIR=$(mktemp -d -p .)
    python3 -m venv --copies --system-site-packages "$env"
    activate_venv
    printf "Created virtual environment %s\n" "$env"
    printf "Upgrading installation tools\n"
    python3 -m pip install setuptools pip wheel --upgrade
    printf "Installing project\n"
    python3 -m pip install .
    rm -rf $TMPDIR && unset TMPDIR
    unlink "$env"/lib64
    sed -i "/PS1=/d" "$env"/bin/activate
    #for file in $NAME/bin/*; do
    #    sed -i '1s/#!.*python$/#!\/usr\/bin\/env python3/' "$file"
    #done
    #sed -i '40s/.*/VIRTUAL_ENV="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}" )")" \&\& pwd)"/' $ENVNAME/bin/activate
}


function source_lcg(){
    printf "Sourcing %s for achitecture %s\n" "${LCG_VIEW}"  "${LCG_ARCH}"
    source "$LCG_SETUP"
    printf "Python version is '%s'\n" "$(python3 --version)"
}

function startup_with_lcg(){
    local env=${env_configs[$1,venv]}
    source_lcg
    if [[ ! -d $env ]]; then
        printf "Virtual environment does not exist, creating virtual environment\n"
        create_venv $env
    else
        activate_venv $env
    fi
}

function rcmode(){
    K="\[\033[0;30m\]"    # black
    R="\[\033[0;31m\]"    # red
    G="\[\033[0;32m\]"    # green
    Y="\[\033[0;33m\]"    # yellow
    B="\[\033[0;34m\]"    # blue
    M="\[\033[0;35m\]"    # magenta
    C="\[\033[0;36m\]"    # cyan
    W="\[\033[0;37m\]"    # white
    EMK="\[\033[1;30m\]"
    EMR="\[\033[1;31m\]"
    EMG="\[\033[1;32m\]"
    EMY="\[\033[1;33m\]"
    EMB="\[\033[1;34m\]"
    EMM="\[\033[1;35m\]"
    EMC="\[\033[1;36m\]"
    EMW="\[\033[1;37m\]"
    BGK="\[\033[40m\]"
    BGR="\[\033[41m\]"
    BGG="\[\033[42m\]"
    BGY="\[\033[43m\]"
    BGB="\[\033[44m\]"
    BGM="\[\033[45m\]"
    BGC="\[\033[46m\]"
    BGW="\[\033[47m\]"
    NONE="\[\033[0m\]"    # unsets color to term's fg color
    local env=${env_configs[$1,venv]}
    if [[ ! -d $env ]]; then
        printf "Virtual environment does not exist, creating virtual environment\n"
        create_venv $env
    else
        activate_venv $env
    fi

    [ -z "$PS1" ] && return

    PS1="${R}[APPTAINER\$( [[ ! -z \${VIRTUAL_ENV} ]] && echo "/\${VIRTUAL_ENV##*/}")]${M}[\t]${W}\u@${C}\h:${G}[\w]> ${NONE}"

    unset PROMPT_COMMAND

    HISTSIZE=50000
    HISTFILESIZE=20000
    export HISTCONTROL="erasedups:ignoreboth"
    export HISTTIMEFORMAT='%F %T '
    export HISTIGNORE=:"&:[ ]*:exit:ls:bg:fg:history:clear"
    shopt -s histappend
    shopt -s cmdhist &>/dev/null
    export HISTFILE=~/.bash_eternal_history
    export CONDOR_CONFIG="/srv/.condor_config"
    export JUPYTER_PATH=/srv/.jupyter
    export JUPYTER_RUNTIME_DIR=/srv/.local/share/jupyter/runtime
    export JUPYTER_DATA_DIR=/srv/.local/share/jupyter
    export IPYTHONDIR=/srv/.ipython
}

function startup_with_container(){
    local in_apptainer=${APPTAINER_COMMAND:-false}
    local container=${env_configs[$1,container]}
    printf "Running in container mode %s\n" "$container"
    if [ "$in_apptainer"  = false ]; then
        printf "Not in apptainer, running command to start container %s.\n" "${container}";
        grep -v '^include' /etc/condor/config.d/01_cmslpc_interactive > .condor_config

        local command="apptainer exec --bind /uscmst1b_scratch/ --bind $HOME/.bash_eternal_history:/srv/.bash_eternal_history --bind ${X509_USER_PROXY%/*}  --bind /cvmfs --home ${PWD}:/srv --pwd /srv "$CONTAINER" /bin/bash --rcfile <(printf \"source setup.sh bashrc\")"
        
        printf "Apptainer command is:\n%s\n" "$command"
        apptainer exec \
                  --bind /uscmst1b_scratch/ \
                  --bind $HOME/.bash_eternal_history:/srv/.bash_eternal_history \
                  --bind /cvmfs \
                  --bind ${X509_USER_PROXY%/*} \
                  --home ${PWD}:/srv \
                  --pwd /srv "$container" /bin/bash \
                  --rcfile <(printf "source setup.sh '$1' bashrc")
        #--rcfile containerrc.sh
    else
        printf "Already in apptainer\n"
    fi
}

function main(){
    local config="$1"
    local mode="${2:-apptainer}"
    if [[ -z ${env_configs[$config,venv]} ]]; then
        printf "Not a valid environment %s\n" "$config"
        return 1
    fi
    case "$mode" in
        apptainer )
            startup_with_container $config
            ;;
        bashrc )
            rcmode $config
            ;;
        * )
            printf "Unknown mode\n"
            return 1
            ;;
    esac
}

main "$@"
