from functools import wraps

def scaled_lumi(lumi_target, lumi_sample):
    def decorator(func):
        @wraps(func)
        def retFunc(fname):
            x = func(fname) * lumi_target / lumi_sample
            return x

        return retFunc

    return decorator


def translate_files(mapping):
    def decorator(func):
        @wraps(func)
        def retFunc(fname, *args, **kwargs):
            x = func(mapping[fname], *args, **kwargs)
            return x

        return retFunc

    return decorator


def scaleQCD(fname):
    SFs = {
        # 2016preVFP
        (0, 100): 19.5 * 1.122e06 * 1e3 / 17657456,
        (0, 200): 19.5 * 8.006e04 * 1e3 / 8886507,
        (0, 300): 19.5 * 1.672e04 * 1e3 / 4978755,
        (0, 500): 19.5 * 1.496e03 * 1e3 / 4433560,
        (0, 700): 19.5 * 3.001e02 * 1e3 / 979344,
        (0, 1000): 19.5 * 4.768e01 * 1e3 / 591966,
        (0, 1500): 19.5 * 4.037e00 * 1e3 / 675657,
        (0, 2000): 19.5 * 6.951e-01 * 1e3 / 668223,
        # 2016postVFP
        (1, 100): 16.8 * 1.124e06 * 1e3 / 19202473,
        (1, 200): 16.8 * 8.040e04 * 1e3 / 9328147,
        (1, 300): 16.8 * 1.668e04 * 1e3 / 5612374,
        (1, 500): 16.8 * 1.502e03 * 1e3 / 4616176,
        (1, 700): 16.8 * 2.995e02 * 1e3 / 903293,
        (1, 1000): 16.8 * 4.756e01 * 1e3 / 663922,
        (1, 1500): 16.8 * 4.024e00 * 1e3 / 698469,
        (1, 2000): 16.8 * 6.963e-01 * 1e3 / 684942,
        # 2017
        (2, 100): 41.5 * 1.125e06 * 1e3 / 37427427,
        (2, 200): 41.5 * 8.013e04 * 1e3 / 19844424,
        (2, 300): 41.5 * 1.669e04 * 1e3 / 11312350,
        (2, 500): 41.5 * 1.506e03 * 1e3 / 10203561,
        (2, 700): 41.5 * 2.998e02 * 1e3 / 1881618,
        (2, 1000): 41.5 * 4.771e01 * 1e3 / 1385631,
        (2, 1500): 41.5 * 4.016e00 * 1e3 / 1458069,
        (2, 2000): 41.5 * 6.979e-01 * 1e3 / 1408971,
        # 2018
        (3, 100): 59.8 * 1.121e06 * 1e3 / 36118282,
        (3, 200): 59.8 * 8.015e04 * 1e3 / 18462183,
        (3, 300): 59.8 * 1.674e04 * 1e3 / 11197722,
        (3, 500): 59.8 * 1.496e03 * 1e3 / 9246898,
        (3, 700): 59.8 * 3.000e02 * 1e3 / 1844165,
        (3, 1000): 59.8 * 4.755e01 * 1e3 / 1330829,
        (3, 1500): 59.8 * 4.030e00 * 1e3 / 1431254,
        (3, 2000): 59.8 * 6.984e-01 * 1e3 / 1357334,
    }
    if "preVFP" in fname:
        period = 0
    elif "UL16" in fname:
        period = 1
    elif "UL17" in fname:
        period = 2
    elif "UL18" in fname:
        period = 3
    else:
        print("ERROR: Data period could not be determined for {}".format(fname))

    if "100to200" in fname:
        HT = 100
    elif "200to300" in fname:
        HT = 200
    elif "300to500" in fname:
        HT = 300
    elif "500to700" in fname:
        HT = 500
    elif "700to1000" in fname:
        HT = 700
    elif "1000to1500" in fname:
        HT = 1000
    elif "1500to2000" in fname:
        HT = 1500
    elif "2000toInf" in fname:
        HT = 2000
    else:
        print("ERROR: HT range could not be determined for {}".format(fname))

    SF = SFs[(period, HT)]
    return SF


def scaleTT(fname):
    lumiTarget = 137.62
    lumiSample = 331506194 / (831.8 * 0.457) * 1e-3
    SF = lumiTarget / lumiSample
    return SF


# -------------------------------------------------
# Signal
# -------------------------------------------------


def scaleSignal(fname):
    points = [
        "1000_400",
        "1000_900",
        "1500_600",
        "1500_1400",
        "2000_900",
        "2000_1900",
        "1000_600",
        "1500_400",
        "2000_400",
        "2000_1400",
        "1500_900",
        "2000_600",
    ]

    lumiTarget = 59.8
    NEventsGen = 10000
    lambdapp312 = 0.1

    SFs = {
        1000: lumiTarget * 47000 * lambdapp312**2 / NEventsGen,
        1500: lumiTarget * 7200 * lambdapp312**2 / NEventsGen,
        2000: lumiTarget * 1600 * lambdapp312**2 / NEventsGen,
    }

    mStop = int(fname.split("_")[0])
    return SFs[mStop]


def scaleZQQ(fname):
    SFs = {
        # 2018
        (3, 200): 59.8 * 1012.0 * 1e3 / 15002757,
        (3, 400): 59.8 * 114.2 * 1e3 / 13930474,
        (3, 600): 59.8 * 25.34 * 1e3 / 12029507,
        (3, 800): 59.8 * 12.99 * 1e3 / 9681521,
    }
    if "preVFP" in fname:
        period = 0
    elif "UL16" in fname:
        period = 1
    elif "UL17" in fname:
        period = 2
    elif "UL18" in fname:
        period = 3
    else:
        print("ERROR: Data period could not be determined for {}".format(fname))

    if "200to400" in fname:
        HT = 200
    elif "400to600" in fname:
        HT = 400
    elif "600to800" in fname:
        HT = 600
    elif "800toInf" in fname:
        HT = 800
    else:
        print("ERROR: HT range could not be determined for {}".format(fname))

    return SFs[(period, HT)]


def scaleZNuNu(fname):
    SFs = {
        # 2018
        (3, 100): 59.8 * 267.0 * 1.1347 * 1e3 / 28876062,
        (3, 200): 59.8 * 73.08 * 1.1347 * 1e3 / 22749608,
        (3, 400): 59.8 * 9.921 * 1.1347 * 1e3 / 19676607,
        (3, 600): 59.8 * 2.409 * 1.1347 * 1e3 / 5968910,
        (3, 800): 59.8 * 1.078 * 1.1347 * 1e3 / 2129122,
        (3, 1200): 59.8 * 0.2514 * 1.1347 * 1e3 / 381695,
        (3, 2500): 59.8 * 0.005614 * 1.1347 * 1e3 / 268224,
    }

    if "preVFP" in fname:
        period = 0
    elif "UL16" in fname:
        period = 1
    elif "UL17" in fname:
        period = 2
    elif "UL18" in fname:
        period = 3
    else:
        print("ERROR: Data period could not be determined for {}".format(fname))

    if "100To200" in fname:
        HT = 100
    elif "200To400" in fname:
        HT = 200
    elif "400To600" in fname:
        HT = 400
    elif "600To800" in fname:
        HT = 600
    elif "800To1200" in fname:
        HT = 800
    elif "1200To2500" in fname:
        HT = 1200
    elif "2500ToInf" in fname:
        HT = 2500
    else:
        print("ERROR: HT range could not be determined for {}".format(fname))

    SF = SFs[(period, HT)]
    return SF


def scaleWQQ(fname):
    SFs = {
        # 2018
        (3, 200): 59.8 * 2549.0 * 1e3 / 14494966,
        (3, 400): 59.8 * 276.5 * 1e3 / 9335298,
        (3, 600): 59.8 * 59.25 * 1e3 / 13633226,
        (3, 800): 59.8 * 28.75 * 1e3 / 13581343,
    }

    if "preVFP" in fname:
        period = 0
    elif "UL16" in fname:
        period = 1
    elif "UL17" in fname:
        period = 2
    elif "UL18" in fname:
        period = 3
    else:
        print("ERROR: Data period could not be determined for {}".format(fname))

    if "200to400" in fname:
        HT = 200
    elif "400to600" in fname:
        HT = 400
    elif "600to800" in fname:
        HT = 600
    elif "800toInf" in fname:
        HT = 800
    else:
        print("ERROR: HT range could not be determined for {}".format(fname))

    SF = SFs[(period, HT)]
    return SF


def scaleDiboson(fname):
    SFs = {
        # 2018
        (3, "WW"): 59.8 * 118.7 * 1e3 / 15679000,
        (3, "WZ"): 59.8 * 47.13 * 1e3 / 7940000,
        (3, "ZZ"): 59.8 * 16.523 * 1e3 / 3526000,
    }

    if "preVFP" in fname:
        period = 0
    elif "UL16" in fname:
        period = 1
    elif "UL17" in fname:
        period = 2
    elif "UL18" in fname:
        period = 3
    else:
        print("ERROR: Data period could not be determined for {}".format(fname))

    if "WW" in fname:
        mode = "WW"
    elif "WZ" in fname:
        mode = "WZ"
    elif "ZZ" in fname:
        mode = "ZZ"
    else:
        print("ERROR: HT range could not be determined for {}".format(fname))

    SF = SFs[(period, mode)]
    return SF


def scaleST(fname):
    SFs = {
        # 2018
        (3, "s-channel"): 59.8 * 11.03 * 0.457 * 1e3 / 10592646,
        (3, "t-channel_antitop"): 59.8 * 80.95 * 1e3 / 90022642,
        (3, "t-channel_top"): 59.8 * 136.02 * 1e3 / 167111718,
        (3, "tW_antitop"): 59.8 * 35.85 * 1e3 / 7748690,
        (3, "tW_top"): 59.8 * 35.85 * 1e3 / 7955614,
    }
    if "preVFP" in fname:
        period = 0
    elif "UL16" in fname:
        period = 1
    elif "UL17" in fname:
        period = 2
    elif "UL18" in fname:
        period = 3
    else:
        print("ERROR: Data period could not be determined for {}".format(fname))

    if "s-channel" in fname:
        mode = "s-channel"
    elif "t-channel_antitop" in fname:
        mode = "t-channel_antitop"
    elif "t-channel_top" in fname:
        mode = "t-channel_top"
    elif "tW_antitop" in fname:
        mode = "tW_antitop"
    elif "tW_top" in fname:
        mode = "tW_top"
    else:
        print("ERROR: HT range could not be determined for {}".format(fname))

    if period == -1 or mode == -1:
        sys.exit()
    SF = SFs[(period, mode)]
    return SF



scale_factor_map = {
    "QCD_bEnriched": scaleQCD,
    "TTToHadronic": scaleTT,
    "ZJetsToQQ": scaleZQQ,
    "WJetsToQQ": scaleWQQ,
    "WW_TuneCP5": scaleDiboson,
    "ZZ_TuneCP5": scaleDiboson,
    "WZ_TuneCP5": scaleDiboson,
    "ST": scaleST,
    "ZJetsToNuNu": scaleZNuNu,
    "signal": scaleSignal,
}


def getFileWeight(fname, scale_lum=False ):
    print("HERE")
    print(fname)
    name,func = next((x,y) for x,y in scale_factor_map.items() if x in fname)
    if scale_lum:
        func = scaled_lumi(137.62, 59.8)(func)
    return func(fname)





