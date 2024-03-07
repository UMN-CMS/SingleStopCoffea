from rich import print

import analyzer as anaz
import analyzer.run_analysis as ra


def main():
    # files = ir.files(anaz)
    t = ra.createPackageZip()
    print(t)


if __name__ == "__main__":
    main()
