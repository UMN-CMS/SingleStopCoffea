from analyzer.datasets import loadSamplesFromDirectory

def getDatasetStyles(directory):
    manager = loadSamplesFromDirectory(directory)
    style_map  = {s : manager[s].getStyle() for s in manager}
    return style_map

