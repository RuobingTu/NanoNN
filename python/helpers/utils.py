import math
import logging
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

def clip(value, lower, upper):
    return lower if value < lower else upper if value > upper else value


def deltaPhi(phi1, phi2):
    try:
        dphi = phi1 - phi2
    except TypeError:
        dphi = phi1.phi - phi2.phi
    while dphi > math.pi:
        dphi -= 2 * math.pi
    while dphi < -math.pi:
        dphi += 2 * math.pi
    return dphi


def deltaR2(eta1, phi1, eta2=None, phi2=None):
    if eta2 is None:
        a, b = eta1, phi1
        return deltaR2(a.eta, a.phi, b.eta, b.phi)
    else:
        deta = eta1 - eta2
        dphi = deltaPhi(phi1, phi2)
        return deta * deta + dphi * dphi
'''
def get_mini_chi2(candi_list,jets):
    minilist = []
    minichi2 = 1000000
    if len(candi_list)==2:
        if jets[candi_list[0]].hadronFlavour == jets[candi_list[1]].hadronFlavour:
            minilist1 = candi_list[0]
            minichi2_1 = (jets[candi_list[0]].M()-125.0)**2
            minilist2 = candi_list[1]
            minichi2_2 = (jets[candi_list[1]].M()-125.0)**2
            if minichi2_2> minichi2_1:
                return [minilist1, minichi2_1, minilist2, minichi2_2]
            else:
                return [minilist2, minichi2_2, minilist1, minichi2_1]
        else:
            minilist = candi_list
            minichi2 = ((jets[candi_list[0]]+jets[candi_list[1]]).M()-125.0)**2
            return [minilist,minichi2]
    elif len(candi_list)==1:
        minilist = candi_list
        minichi2 = (jets[candi_list[0]].M()-125.0)**2
        return [minilist,minichi2]
    elif len(candi_list)==0:
        minilist = candi_list
        minichi2 = 1000000
        return [minilist,minichi2]
    else:
        new_list = [[x,y] for x in candi_list for y in candi_list if (x!=y and jets[x].hadronFlavour != jets[y].hadronFlavour)]
        for tmp_list in new_list:
            tmp_chi2 = ((jets[tmp_list[0]]+jets[tmp_list[1]]).M()-125.0)**2
            if tmp_chi2<minichi2:
                minichi2 = tmp_chi2
                minilist = tmp_list
        return [minilist,minichi2]
'''

def get_mini_chi2(candi_list,jets):
    minilist = []
    minichi2 = 1000000
    if len(candi_list)==2:
        minilist = candi_list
        minichi2 = ((jets[candi_list[0]]+jets[candi_list[1]]).M()-125.0)**2
        return [minilist,minichi2]
    elif len(candi_list)==1:
        minilist = candi_list
        minichi2 = (jets[candi_list[0]].M()-125.0)**2
        return [minilist,minichi2]
    elif len(candi_list)==0:
        minilist = candi_list
        minichi2 = 1000000
        return [minilist,minichi2]
    else:
        new_list = [[x,y] for x in candi_list for y in candi_list if x!=y ]
        for tmp_list in new_list:
            tmp_chi2 = ((jets[tmp_list[0]]+jets[tmp_list[1]]).M()-125.0)**2
            if tmp_chi2<minichi2:
                minichi2 = tmp_chi2
                minilist = tmp_list
        return [minilist,minichi2]

def fj_get_mini_chi2(candi_list,jets):
    mini_idx = -1 
    minichi2 = 1000000
    if len(candi_list)==0:
        mini_idx = -1
        minichi2 = 1000000
        return [mini_idx,minichi2]
    else:
        for tmp_list in candi_list:
            tmp_chi2 = (jets[tmp_list].PNmass - 125)**2
            if tmp_chi2<minichi2:
                minichi2 = tmp_chi2
                mini_idx = tmp_list
        return [mini_idx,minichi2]


def deltaR(eta1, phi1, eta2=None, phi2=None):
    return math.sqrt(deltaR2(eta1, phi1, eta2, phi2))


def closest(obj, collection, presel=lambda x, y: True):
    ret = None
    retidx = None
    dr2Min = 1e6
    for idx,x in enumerate(collection):
        if not presel(obj, x):
            continue
        dr2 = deltaR2(obj, x)
        if dr2 < dr2Min:
            ret = x
            retidx = idx
            dr2Min = dr2
    return (ret, math.sqrt(dr2Min), retidx)


def polarP4(obj=None, pt='pt', eta='eta', phi='phi', mass='mass'):
    if obj is None:
        return ROOT.Math.PtEtaPhiMVector()
    pt_val = getattr(obj, pt) if pt else 0
    eta_val = getattr(obj, eta) if eta else 0
    phi_val = getattr(obj, phi) if phi else 0
    mass_val = getattr(obj, mass) if mass else 0
    return ROOT.Math.PtEtaPhiMVector(pt_val, eta_val, phi_val, mass_val)


def p4(obj=None, pt='pt', eta='eta', phi='phi', mass='mass'):
    v = polarP4(obj, pt, eta, phi, mass)
    return ROOT.Math.XYZTVector(v.px(), v.py(), v.pz(), v.energy())


def sumP4(*args):
    p4s = [polarP4(x) for x in args]
    return sum(p4s, ROOT.Math.PtEtaPhiMVector())


def p4_str(p):
    return '(pt=%s, eta=%s, phi=%s, mass=%s)' % (p.pt(), p.eta(), p.phi(), p.mass())


def get_subjets(jet, subjetCollection, idxNames=('subJetIdx1', 'subJetIdx2')):
    subjets = []
    for idxname in idxNames:
        idx = getattr(jet, idxname)
        if idx >= 0:
            subjets.append(subjetCollection[idx])
    subjets = sorted(subjets, key=lambda x: x.pt, reverse=True)  # sort by pt
    return subjets


def corrected_svmass(sv):
    pproj = polarP4(sv).P() * math.sin(sv.pAngle)
    return math.sqrt(sv.mass * sv.mass + pproj * pproj) + pproj


def transverseMass(obj, met):
    cos_dphi = math.cos(deltaPhi(obj, met))
    return math.sqrt(2 * obj.pt * met.pt * (1 - cos_dphi))


def minValue(collection, fallback=99):
    if len(collection) == 0:
        return fallback
    else:
        return min(collection)


def maxValue(collection, fallback=0):
    if len(collection) == 0:
        return fallback
    else:
        return max(collection)


def configLogger(name, loglevel=logging.INFO, filename=None):
    # define a Handler which writes INFO messages or higher to the sys.stderr
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    console = logging.StreamHandler()
    console.setLevel(loglevel)
    console.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
    logger.addHandler(console)
    if filename:
        logfile = logging.FileHandler(filename)
        logfile.setLevel(loglevel)
        logfile.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
        logger.addHandler(logfile)

def get_bjet_sf(pt, eta):
    """
    This function returns the b-jet scale factor.
    A real implementation would read these values from a data file.
    """
    if pt > 670:
        pt = 670
    if abs(eta) > 2.4:
        return 1.0

    return 0.938887 + 0.00017124 * pt - 1.55395e-07 * pt * pt
