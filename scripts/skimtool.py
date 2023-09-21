from XRootD import client
from XRootD.client.flags import DirListFlags, OpenFlags, MkDirFlags, QueryCode
import argparse
from urllib.parse import urlparse, urlunparse, urljoin
from analyzer.datasets import loadSamplesFromDirectory
import re
from yaml import load, dump
import uproot

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import sys


def countEvents(path):
    with uproot.open({path: None}, timeout=20) as f:
        tree = f["Events"]
        return tree.num_entries


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def listFiles(server, directory):
    myclient = client.FileSystem(server)
    status, listing = myclient.dirlist(directory, DirListFlags.STAT)
    return [urljoin(directory, x.name) for x in listing]


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Generate metadata for skimmed dataset collections derived from existing ones"
    )
    parser.add_argument("-p", "--protocol", default="root", type=str, help="Protocol")
    parser.add_argument("-s", "--server", required=True, type=str, help="Server")
    parser.add_argument("-d", "--directory", required=True, type=str, help="Directory")
    parser.add_argument("-o", "--output", required=True, type=str, help="Output")
    parser.add_argument("--prefix", required=True, type=str, help="Output")
    parser.add_argument(
        "-c", "--collection", required=True, type=str, help="Target Collection"
    )
    args = parser.parse_args()
    return args


def doMatching(files, sets, setname_transform):
    ret = {}
    for ss in sets:
        sname = ss.name
        transformed_name = setname_transform(sname)
        eprint(f"{sname} => {transformed_name}")
        matched = [x for x in files if transformed_name in x]
        ret[sname] = matched
    return ret


def repeatSub(string, sublist):
    for x in sublist:
        string = re.sub(x[0], x[1], string)
    return string


default_subs = {
    "QCDInclusive2018": [("QCDInclusive2018", "QCD")],
    "WJetsToQQ2018": [("2018", ""), (f"HT-(\d+)To(\d+|Inf)", r"HT-\1to\2")],
    "ZJetsToQQ2018": [("2018", ""), (f"HT-(\d+)To(\d+|Inf)", r"HT-\1to\2")],
    "ZJetsToNuNu2018": [("2018", ""), ("HT", "HT-")],
    "Diboson2018": [("2018", ""), (f"(..)2018", r"\1_TuneCP5_13TeV")],
    "TTToHadronic2018": [("TTToHadronic2018", "TT")],
    "STHadronic2018": [
        ("STHadronic2018_", "ST_"),
        (r"SChannel", r"s-channel"),
        (r"TChannel", r"t-channel"),
        (f"AntiTop", "_antitop"),
        (f"Top", f"_top"),
        (f"TW", "tW"),
    ],
}


def main():
    args = parseArgs()
    url = urlunparse((args.protocol, args.server, args.directory, "", "", ""))
    scheme, server, *_ = urlparse(url)
    prot_server = f"{scheme}://{server}"

    manager = loadSamplesFromDirectory("datasets")

    files = listFiles(prot_server, args.directory)
    iscol = False
    if args.collection in manager.collections:
        iscol = True
        collection = manager.collections[args.collection]
        cname = collection.name
        allsets = collection.sets
    else:
        collection = manager.sets[args.collection]
        cname = collection.name
        allsets = [collection]

    sublist = default_subs.get(cname, [])
    matching = doMatching(files, allsets, lambda x: repeatSub(x, sublist))
    matching = {
        name: [f"root://cmsxrootd.fnal.gov/{f}" for f in files]
        for name, files in matching.items()
    }
    structs = []
    if iscol:
        structs = [
            dict(
                name=f"{args.prefix}_{cname}",
                derived_from=cname,
                sets=[f"{args.prefix}_{x}" for x in matching],
            )
        ]
    structs += [
        dict(
            name=f"{args.prefix}_{name}",
            derived_from=name,
            files=f,
            n_events=sum(countEvents(x) for x in f),
        )
        for name, f in matching.items()
    ]
    print(dump(structs, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
