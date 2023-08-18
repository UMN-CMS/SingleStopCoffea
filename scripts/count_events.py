import uproot
import argparse 

class UprootMissTreeError(uproot.exceptions.KeyInFileError):
    pass
        
def main():
    parser = argparse.ArgumentParser("Count events in root file")
    parser.add_argument(
        "filename",
        help="Path to the file",
    )

    args = parser.parse_args()
    with uproot.open({args.filename: None}, timeout=20) as f:
        tree = f["Events"]
        print(tree.num_entries)


if __name__ == "__main__":
    main()
