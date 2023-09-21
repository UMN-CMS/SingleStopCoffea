from pathlib import Path
from urllib.parse import urlparse, urlunparse
from contextlib import contextmanager
import shutil



def copyFile(fr, to):
    fr_scheme, fr_netloc, fr_path, *fr_rest = urlparse(str(fr))
    to_scheme, to_netloc, to_path, *to_rest = urlparse(str(to))
    if not fr_scheme:
        fr_path = str(Path(fr_path).resolve().absolute()) 
    if not to_scheme:
        to_path = str(Path(to_path).resolve().absolute()) 
    fr = urlunparse((fr_scheme, fr_netloc, fr_path, *fr_rest))
    to = urlunparse((to_scheme, to_netloc, to_path, *to_rest))
    xrootd = any(x == 'root' for x in (fr_scheme , to_scheme))
    if xrootd:
        import XRootD
        import XRootD.client
        copyproc = XRootD.client.CopyProcess()
        copyproc.add_job(str(fr), str(to))
        copyproc.prepare()
        copyproc.run()
        client = XRootD.client.FileSystem(to_netloc)
        status = client.locate(to_path, XRootD.client.flags.OpenFlags.READ)
        assert status[0].ok
        del client
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

def getStem(url):
    scheme, netloc, path, *fr_rest = urlparse(str(url))
    return str(Path(path).stem)
    
