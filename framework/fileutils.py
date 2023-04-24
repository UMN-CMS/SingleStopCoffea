from pathlib import Path
from urllib.parse import urlparse, urlunparse
from contextlib import contextmanager



def copyFile(fr, to):
    fr_scheme, fr_netloc, fr_path, *rest = urlparse(str(fr))
    to_scheme, to_netloc, to_path, *rest = urlparse(str(to))
    xrootd = any(x == 'root' for x in (fr_scheme , to_scheme))
    if xrootd:
        import XRootD
        import XRootD.client
        copyproc = XRootD.client.CopyProcess()
        copyproc.add_job(str(fr), str(to))
        copyproc.prepare()
        copyproc.run()
        del copyproc
    else:
        to = Path(to)
        if not to.parent.is_dir():
            to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fr, to)

def appendToUrl(url,*args):
    scheme, netloc, path , *rest = urlparse(str(url))
    path = Path(path, *args)
    return urlunparse((scheme, netloc, str(path), *rest))
    

if __name__ == "__main__" or True:
    url = "root://cmseos.fnal.gov//store/user/ckapsiak/test/"
    x = appendToUrl(url , "other", "andmore")
    print(x)
