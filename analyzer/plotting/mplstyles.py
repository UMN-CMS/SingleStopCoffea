import importlib.resources as imp_res
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

from . import static


def loadStyles():
    HAS_IMPRES = hasattr(imp_res, "files")
    if HAS_IMPRES:
        style = imp_res.files(static) / "style.mplstyle"
        font_dirs = [imp_res.files(static) / "fonts"]
    else:
        from pkg_resources import (resource_filename, resource_listdir,
                                   resource_string)

        style = resource_filename("analyzer.plotting.static", "style.mplstyle")
        font_dirs = [resource_filename("analyzer.plotting.static", "fonts")]

    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font in font_files:
        font_manager.fontManager.addfont(font)
    plt.style.use(style)
