import uproot
import hist
import pickle

def main():
    file1 = "f2.pkl"
    file2 = "f1.root"

    data_1 = pickle.load(open(file2, 'rb'))
    data_2 = uproot.open(file2)

