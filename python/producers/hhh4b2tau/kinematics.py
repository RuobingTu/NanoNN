"""Pure transverse-kinematics helpers used by the v27 producer.

Free functions with no producer state -- moved verbatim.
"""
import os
import itertools
import ROOT
import random
ROOT.PyConfig.IgnoreCommandLineOptions = True
import numpy as np
from collections import Counter
from operator import itemgetter
import math
import onnxruntime


from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from PhysicsTools.NanoNN.helpers.jetmetCorrector import JetMETCorrector, rndSeed
from PhysicsTools.NanoNN.helpers.triggerHelper import passTrigger
from PhysicsTools.NanoNN.helpers.utils import closest, sumP4, polarP4, configLogger, get_subjets, deltaPhi, deltaR, get_mini_chi2, fj_get_mini_chi2, getgplist
from PhysicsTools.NanoNN.helpers.nnHelper import convert_prob, ensemble
from PhysicsTools.NanoNN.helpers.massFitter import fitMass
from PhysicsTools.NanoNN.helpers.btagWeightCalculator import BTagWeightCalculator
try:
    import correctionlib
except ImportError:
    correctionlib = None

import logging
logger = logging.getLogger('nano')
configLogger('nano', loglevel=logging.INFO)

UNDEFINED = -999.0


def _mt_sq(pt_a, phi_a, pt_b, phi_b):
    """Squared transverse mass of two massless transverse vectors."""
    return max(2.0 * pt_a * pt_b * (1.0 - math.cos(phi_a - phi_b)), 0.0)


def transverse_mass(pt_a, phi_a, pt_b, phi_b):
    """mT = sqrt(2 pT_a pT_b (1 - cos dphi)). Same convention as lep{i}Mt."""
    return math.sqrt(_mt_sq(pt_a, phi_a, pt_b, phi_b))


def mt_tot(pt1, phi1, pt2, phi2, met, metphi):
    """Total transverse mass of a two-leg + MET system (standard H->tautau variable).

    mT_tot = sqrt(mT^2(l1,MET) + mT^2(l2,MET) + mT^2(l1,l2))
    """
    return math.sqrt(_mt_sq(pt1, phi1, met, metphi)
                     + _mt_sq(pt2, phi2, met, metphi)
                     + _mt_sq(pt1, phi1, pt2, phi2))


def d_zeta(pt1, phi1, pt2, phi2, met, metphi, alpha=0.85):
    """Return (Dzeta, pzeta_vis, pzeta_miss) for two visible legs plus MET.

    The zeta axis is the bisector of the two legs' transverse UNIT vectors --
    using unit vectors (not momenta) is what makes the bisector independent of
    the pT balance between the legs, and is the standard CDF/CMS definition.

    Degenerate case: exactly back-to-back legs make the bisector ill-defined;
    those events return UNDEFINED rather than an arbitrary axis.
    """
    u1x, u1y = math.cos(phi1), math.sin(phi1)
    u2x, u2y = math.cos(phi2), math.sin(phi2)
    zx, zy = u1x + u2x, u1y + u2y
    norm = math.hypot(zx, zy)
    if norm < 1e-6:
        return UNDEFINED, UNDEFINED, UNDEFINED
    zx, zy = zx / norm, zy / norm
    pzeta_vis = (pt1 * u1x + pt2 * u2x) * zx + (pt1 * u1y + pt2 * u2y) * zy
    pzeta_miss = met * math.cos(metphi) * zx + met * math.sin(metphi) * zy
    return pzeta_miss - alpha * pzeta_vis, pzeta_vis, pzeta_miss


def mt2_massless(p4_1, p4_2, met, metphi):
    """MT2 of two visible 4-vectors with massless invisible particles.

    Lester-Nachman bisection (arXiv:1411.4312v7), vendored in helpers/mt2_cc/.
    Returns UNDEFINED if the solver reports an error (negative return value).

    Endpoint reminder: MT2 is bounded by the parent mass ONLY when the event
    really is two symmetric chains Y -> visible_i + invisible_i and the trial
    invisible mass equals the true one. With m_chi = 0 that holds for
    W -> l nu pairs (endpoint m_W), which is the ttbar-dilepton case. It does
    NOT hold for a single lost-lepton neutrino, so mt2_bb in the lepton-vetoed
    1tau0l channel is a bounded MET-angular discriminant, not an m_top endpoint.

    IMPORTANT degeneracy (verified numerically against a brute-force 2-D scan):
    with massless visibles AND m_chi = 0, MT2 is EXACTLY 0 whenever MET lies
    inside the angular wedge spanned by the two legs. In that case
    MET = a*p1hat + b*p2hat with a,b >= 0, so both legs can be handed collinear
    invisible momenta and both mT vanish. Consequences:
      * H->tautau signal has its neutrinos collinear with the visible legs, so
        MET sits inside the wedge and mt2_ll piles up at exactly 0 -- expect a
        large delta-function at zero, it is not a bug.
      * All the separation therefore lives in the non-zero ttbar/W tail, and
        mt2_ll is strongly correlated with Dzeta, which measures the same
        "is MET inside the wedge" geometry continuously. Check that they are not
        redundant before spending input slots on both.
      * If a less degenerate variable is wanted, re-run with a non-zero trial
        mass (m_chi ~ m_W) -- that is a one-line change here, but note the
        endpoint interpretation changes with it.
    """
    val = ROOT.asymm_mt2_lester_bisect.get_mT2(
        p4_1.M(), p4_1.Px(), p4_1.Py(),
        p4_2.M(), p4_2.Px(), p4_2.Py(),
        met * math.cos(metphi), met * math.sin(metphi),
        0.0, 0.0)
    return val if val >= 0 else UNDEFINED


def event_shapes(p4list):
    """Sphericity-tensor event shapes from a list of 4-vectors (lab frame).

    Returns a dict with both tensor conventions:

    Quadratic tensor  S^ab = sum p^a p^b / sum |p|^2  (pT-weighted, the classic
    LEP definition). Eigenvalues l1>=l2>=l3 give
        sphericity = 3/2 (l2+l3),  aplanarity = 3/2 l3,  planarity = l2 - l3.

    Linear (linearised) tensor  L^ab = sum p^a p^b/|p| / sum |p|  is infrared
    safe -- it is not dominated by the single hardest object, so it is the more
    stable choice in busy QCD events. Its eigenvalues give the standard
        C = 3 (l1 l2 + l2 l3 + l3 l1),  D = 27 l1 l2 l3,
    plus a linearised sphericity 3/2 (l2+l3) for direct comparison.

    Fewer than two objects leaves the tensor rank-deficient and the shapes
    meaningless, so those events get UNDEFINED.
    """
    out = {'sphericity': UNDEFINED, 'aplanarity': UNDEFINED, 'planarity': UNDEFINED,
           'sphericity_lin': UNDEFINED, 'shapeC': UNDEFINED, 'shapeD': UNDEFINED}
    if len(p4list) < 2:
        return out

    quad = np.zeros((3, 3))
    lin = np.zeros((3, 3))
    norm_quad = 0.0
    norm_lin = 0.0
    for p4 in p4list:
        p = np.array([p4.Px(), p4.Py(), p4.Pz()])
        pmag = float(np.linalg.norm(p))
        if pmag <= 0:
            continue
        outer = np.outer(p, p)
        quad += outer
        norm_quad += pmag * pmag
        lin += outer / pmag
        norm_lin += pmag
    if norm_quad <= 0 or norm_lin <= 0:
        return out

    # Symmetric, positive semi-definite -> eigvalsh gives ascending real eigenvalues.
    lq = np.linalg.eigvalsh(quad / norm_quad)[::-1]      # l1 >= l2 >= l3
    ll = np.linalg.eigvalsh(lin / norm_lin)[::-1]
    lq = np.clip(lq, 0.0, None)
    ll = np.clip(ll, 0.0, None)

    out['sphericity'] = 1.5 * (lq[1] + lq[2])
    out['aplanarity'] = 1.5 * lq[2]
    out['planarity'] = lq[1] - lq[2]
    out['sphericity_lin'] = 1.5 * (ll[1] + ll[2])
    out['shapeC'] = 3.0 * (ll[0] * ll[1] + ll[1] * ll[2] + ll[2] * ll[0])
    out['shapeD'] = 27.0 * ll[0] * ll[1] * ll[2]
    return out

