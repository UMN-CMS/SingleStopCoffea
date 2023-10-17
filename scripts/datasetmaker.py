from XRootD import client
from XRootD.client.flags import DirListFlags, OpenFlags, MkDirFlags, QueryCode
import argparse
from urllib.parse import urlparse, urlunparse, urljoin
import re
from yaml import load, dump
import yaml
from yaml.resolver import BaseResolver
import uproot
import itertools as it
from pathlib import Path

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import sys

from analyzer.datasets import loadSamplesFromDirectory


class SingleQuoted(str):
    pass


def represent_single_quoted(dumper, data):
    return dumper.represent_scalar(BaseResolver.DEFAULT_SCALAR_TAG, data, style="'")


yaml.add_representer(SingleQuoted, represent_single_quoted)


def listFiles(server, directory):
    myclient = client.FileSystem(server)
    status, listing = myclient.dirlist(directory, DirListFlags.STAT)
    return [urljoin(directory, x.name) for x in listing]


def countEvents(path):
    with uproot.open({path: None}, timeout=20) as f:
        tree = f["Events"]
        return tree.num_entries


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Generate metadata for skimmed dataset collections derived from existing ones"
    )
    parser.add_argument(
        "-p", "--protocol", default="root", type=str, help="EOS protocol"
    )
    parser.add_argument("-s", "--server", required=True, type=str, help="EOS server")
    parser.add_argument(
        "-d", "--directory", required=True, type=str, help="EOS directory"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default="-",
        help="Output file, or '-' to print to stdout",
    )
    parser.add_argument(
        "-m",
        "--match-re",
        required=True,
        type=str,
        help="Regex, with capture groups, to match datasets. Two files are condidered to be in the same dataset if their capture groups are equal.",
    )
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        type=str,
        help="Name to use for each dataset, may use '\1', '\2', etc to insert the captured experssion from the '--match-re' option.",
    )
    parser.add_argument(
        "-l",
        "--lumi-re",
        default=None,
        type=str,
        help="",
    )
    parser.add_argument(
        "-t",
        "--title",
        default=None,
        type=str,
        help="Title to use for each dataset, may use '\1', '\2', etc to insert the captured experssion from the '--match-re' option.",
    )
    parser.add_argument(
        "-u",
        "--update",
        default=None,
        type=argparse.FileType("r"),
        help="File to update",
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


def replaceString(s, t):
    for x in range(1, len(t) + 1):
        s = s.replace(f"\\{x}", t[x - 1])
    return s


def getLumiInfo(match, existing):
    f = next((x for x in existing if re.search(match, x["name"])), None)
    if f is None:
        return {}
    else:
        return {"x_sec": f["x_sec"], "lumi": f["lumi"]}


def main():
    args = parseArgs()
    already_present = {}
    if args.update:
        r = load(args.update, Loader=Loader)
        already_present = {x["name"]: x for x in r}
    url = urlunparse((args.protocol, args.server, args.directory, "", "", ""))
    scheme, server, *_ = urlparse(url)
    prot_server = f"{scheme}://{server}"
    files = listFiles(prot_server, args.directory)
    files = [
        f for f in files if re.search(args.match_re, str(Path(urlparse(f)[2]).stem))
    ]
    files = sorted(files)

    groups = it.groupby(
        files,
        lambda x: re.search(args.match_re, str(Path(urlparse(x)[2]).stem)).groups(),
    )
    matching = [
        (
            replaceString(args.name, fields),
            fields,
            [f"root://cmsxrootd.fnal.gov/{f}" for f in files],
        )
        for fields, files in groups
    ]
    matching = [x for x in matching if x[0] not in already_present]

    structs = []
    structs += [
        dict(
            name=name,
            files=f,
            n_events=sum(countEvents(x) for x in f),
            **(
                {"title": SingleQuoted(replaceString(args.title, fields))}
                if args.title
                else {}
            ),
            **(
                getLumiInfo(replaceString(args.lumi_re, fields), already_present.values())
                if args.lumi_re
                else {}
            ),
        )
        for name, fields, f in matching
    ]
    args.output.write(dump(structs, indent=2, sort_keys=False, width=1000))


if __name__ == "__main__":
    main()
