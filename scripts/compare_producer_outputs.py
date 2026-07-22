#!/usr/bin/env python3
"""Refactor gate: compare two producer output ROOT files branch-by-branch.

Default criterion is EXACT equality, per branch and per event. A pure
code-motion refactor has no legitimate reason to change any value, so anything
that is merely "close" is reported for individual investigation rather than
silently accepted. The max relative deviation is printed alongside so an
FP-summation-order explanation can be judged case by case.

Usage:
  compare_outputs.py OLD.root NEW.root [--allow-missing b1,b2,...]

--allow-missing lists branches that are expected to be absent from NEW (the
dead branches deliberately removed); the tool then also asserts they were
genuinely constant-zero in OLD, which is the justification for dropping them.
"""
import argparse
import sys

import numpy as np
import uproot


def load(path, tree='Events'):
    f = uproot.open(path)
    names = sorted({k.split(';')[0] for k in f[tree].keys()})
    return f[tree], names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('old')
    ap.add_argument('new')
    ap.add_argument('--allow-missing', default='')
    ap.add_argument('--tree', default='Events')
    a = ap.parse_args()

    allow = {s for s in a.allow_missing.split(',') if s}

    told, nold = load(a.old, a.tree)
    tnew, nnew = load(a.new, a.tree)

    print('OLD %s  : %d events, %d branches' % (a.old, told.num_entries, len(nold)))
    print('NEW %s  : %d events, %d branches' % (a.new, tnew.num_entries, len(nnew)))

    ok = True
    if told.num_entries != tnew.num_entries:
        print('\n*** EVENT COUNT MISMATCH -- selection changed, gate FAILS')
        return 1
    if told.num_entries == 0:
        print('\n*** OLD file has 0 events -- gate is vacuous, rerun with more events')
        return 1

    removed = set(nold) - set(nnew)
    added = set(nnew) - set(nold)

    unexpected_removed = removed - allow
    if unexpected_removed:
        ok = False
        print('\n*** branches present in OLD but MISSING from NEW (unexpected):')
        for b in sorted(unexpected_removed):
            print('      ', b)
    if added:
        print('\n(branches new in NEW, not compared): %s' % ', '.join(sorted(added)))

    # Deliberately removed branches must have been constant-zero in OLD.
    if allow & removed:
        print('\n=== justification for deliberately removed branches ===')
        arrs = told.arrays(sorted(allow & removed), library='np')
        for b, v in sorted(arrs.items()):
            v = np.asarray(v).astype(float)
            u = np.unique(v)
            good = (len(u) == 1 and u[0] == 0)
            ok &= good
            print('   %-20s nunique=%-4d min=%-8.4g max=%-8.4g %s'
                  % (b, len(u), v.min(), v.max(),
                     'constant zero -> safe to drop' if good else '*** NOT constant zero'))

    common = sorted(set(nold) & set(nnew))
    print('\n=== comparing %d common branches, exact equality required ===' % len(common))

    bad, near = [], []
    for b in common:
        vo = np.asarray(told[b].array(library='np'))
        vn = np.asarray(tnew[b].array(library='np'))
        if vo.dtype == object or vn.dtype == object:      # jagged
            vo = np.concatenate([np.asarray(x).ravel() for x in vo]) if len(vo) else np.array([])
            vn = np.concatenate([np.asarray(x).ravel() for x in vn]) if len(vn) else np.array([])
        if vo.shape != vn.shape:
            bad.append((b, 'shape %s vs %s' % (vo.shape, vn.shape), np.inf))
            continue
        # NaN-aware exact comparison
        if np.issubdtype(vo.dtype, np.floating):
            same = ((vo == vn) | (np.isnan(vo) & np.isnan(vn)))
        else:
            same = (vo == vn)
        if same.all():
            continue
        fo, fn = vo.astype(float), vn.astype(float)
        den = np.maximum(np.abs(fo), 1e-30)
        with np.errstate(invalid='ignore', divide='ignore'):
            rel = np.nanmax(np.abs(fn - fo) / den)
        n_diff = int((~same).sum())
        entry = (b, '%d/%d events differ' % (n_diff, same.size), rel)
        (near if rel < 1e-6 else bad).append(entry)

    if near:
        print('\n--- differ but within 1e-6 relative (investigate, likely FP order) ---')
        for b, d, r in near:
            print('   %-34s %-22s max rel dev %.3g' % (b, d, r))
        ok = False
    if bad:
        print('\n*** DIFFER beyond 1e-6 relative -- real behaviour change ***')
        for b, d, r in bad:
            print('   %-34s %-22s max rel dev %.3g' % (b, d, r))
        ok = False

    print('\n' + '=' * 72)
    if ok and not near and not bad:
        print('GATE PASSED: all %d common branches exactly identical across %d events'
              % (len(common), told.num_entries))
    else:
        print('GATE FAILED')
    print('=' * 72)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
