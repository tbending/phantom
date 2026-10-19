#!/usr/bin/env python3
"""Compare two phantom runs of the same substeps: the GPU path against the CPU path.

    python3 -I gpu_compare_substeps.py <run A dir> <run B dir> [--prefix disc]

Written for individual timesteps on the GPU (GPU_IND=yes), where two runs from one
restart must agree on which particles were active at every substep, not only on the
final state.  B is the reference.  Reports:

  steps    the '> step i / n ... moved m' lines of each run.log (or *.log): the number of
           substeps and the active count of each.  Ints, so they must match exactly; the
           first difference is printed.
  dump     the last full dump of each run, record by record: how many particle arrays are
           byte-identical (Gate 0 needs all of them).
  types    per particle type: rms |v_A - v_B| / rms |v_B|, rms |x_A - x_B| / mean h_B and
           max |h_A/h_B - 1|, over particles alive in both.
  ev       the last row the two .ev files share: the largest relative differences by
           column (scaled by the column's largest magnitude in B).

Exit status 0 if the step sequence matches and every figure is printed; the verdict on
the figures belongs to the caller, against the noise floor of the case.  Needs numpy.
"""
import glob
import os
import re
import struct
import sys

import numpy as np

SIZES = [4, 1, 2, 4, 8, 8, 4, 8]                 # int int1 int2 int4 int8 real real4 real8
DTYPES = ['<i4', '<i1', '<i2', '<i4', '<i8', '<f8', '<f4', '<f8']
EVCOLS = ('dt', 'ekin', 'etherm', 'emag', 'epot', 'etot', 'totmom', 'angtot', 'rho max', 'dust/gas')
TYPENAMES = {1: 'gas', 2: 'empty', 3: 'bound', 4: 'star', 5: 'darkm', 6: 'bulge', 7: 'dust'}


def records(fn):
    d = open(fn, 'rb').read()
    off = 0
    while off + 4 <= len(d):
        n = struct.unpack('<i', d[off:off + 4])[0]
        s = off + 4
        if n < 0 or s + n + 4 > len(d):
            raise ValueError(f"{fn}: bad record marker at byte {off}")
        yield d[s:s + n]
        off = s + n + 4


def read_dump(fn):
    """Return (header dict, list of {tag: array} per block-array), full dumps only."""
    rec = records(fn)
    next(rec)                                     # int, real, int, version, int: format check
    fileid = next(rec).decode(errors='replace').strip()
    if not fileid.startswith('F'):
        raise ValueError(f"{fn} is not a full dump ({fileid[:20]!r})")
    hdr = {}
    for k in range(8):
        nv = struct.unpack('<i', next(rec))[0]
        if nv > 0:
            tags = next(rec)
            vals = np.frombuffer(next(rec), DTYPES[k])
            for j in range(nv):
                hdr.setdefault(tags[16 * j:16 * j + 16].decode().strip(), vals[j])
    ntot = struct.unpack('<i', next(rec))[0]
    blocks = []
    for _ in range(ntot):
        r = next(rec)
        number = struct.unpack('<q', r[:8])[0]
        nums = struct.unpack('<8i', r[8:40])
        blocks.append((number, nums))
    out = []
    for number, nums in blocks:
        arrs = {}
        for k in range(8):
            for _ in range(nums[k]):
                tag = next(rec).decode().strip()
                a = np.frombuffer(next(rec), DTYPES[k])
                if len(a) != number:
                    raise ValueError(f"{fn}: array {tag} has {len(a)} values, expected {number}")
                arrs.setdefault(tag, a)
        out.append(arrs)
    return hdr, out


def last_full_dump(run, prefix):
    pat = f"{run}/{prefix}_[0-9][0-9][0-9][0-9][0-9]" if prefix else f"{run}/*_[0-9][0-9][0-9][0-9][0-9]"
    for f in sorted(glob.glob(pat), reverse=True):
        with open(f, 'rb') as fh:
            head = fh.read(40)                    # format-check record, then the file id
            n = struct.unpack('<i', head[:4])[0]
            if head[n + 12:n + 13] == b'F':
                return f
    return None


def step_lines(run):
    logs = [f for f in sorted(glob.glob(f"{run}/run.log")) + sorted(glob.glob(f"{run}/*.log"))]
    if not logs:
        return None
    pat = re.compile(r'^> step\s+(\d+)\s*/\s*(\d+)\s+t\s*=\s*(\S+)\s+dt\s*=\s*(\S+)\s+moved\s+(\d+)')
    out = []
    for line in open(logs[0], errors='replace'):
        m = pat.match(line)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(5))))
    return out


def ev_table(run, prefix):
    pat = f"{run}/{prefix}*.ev" if prefix else f"{run}/*.ev"
    fs = [f for f in sorted(glob.glob(pat)) if 'Sink' not in os.path.basename(f)]
    if not fs:
        return None, None
    f = fs[-1]
    names = [n.strip() for n in re.findall(r'\[\d+\s+([^\]]+)\]', open(f).readline())]
    # Fortran drops the E of a three-digit exponent (5.35+135); put it back for numpy
    fix = re.compile(r'(?<=[0-9.])([+-][0-9]{3})(?![0-9])')
    rows = [fix.sub(r'E\1', line) for line in open(f) if not line.lstrip().startswith('#')]
    return names, np.atleast_2d(np.loadtxt(rows))


def main(argv):
    args = [a for a in argv if not a.startswith('--')]
    prefix = ''
    for i, a in enumerate(argv):
        if a == '--prefix':
            prefix = argv[i + 1]
            args.remove(prefix)
    A, B = args[0], args[1]
    ok = True

    sa, sb = step_lines(A), step_lines(B)
    if sa is None or sb is None:
        print("steps    no run log in one of the runs")
    else:
        first = next((i for i, (x, y) in enumerate(zip(sa, sb)) if (x[0], x[1], x[3]) != (y[0], y[1], y[3])), None)
        same = first is None and len(sa) == len(sb)
        print(f"steps    {len(sa)} vs {len(sb)} substeps, active counts "
              + ("IDENTICAL" if same else "DIFFER"))
        if not same:
            ok = False
            if first is not None:
                print(f"         first difference at line {first}: A {sa[first]} B {sb[first]}")
        tot = sum(x[3] for x in sb)
        print(f"         particle updates in B: {tot}")

    fa, fb = last_full_dump(A, prefix), last_full_dump(B, prefix)
    if not fa or not fb:
        print("dump     no full dump in one of the runs")
        return 1
    print(f"dump     {os.path.basename(fa)} vs {os.path.basename(fb)}")
    ra, rb = list(records(fa)), list(records(fb))
    big = [k for k, r in enumerate(rb) if len(r) >= 1000]
    nsame = sum(1 for k in big if k < len(ra) and ra[k] == rb[k])
    print(f"         {nsame}/{len(big)} particle records byte-identical"
          + ("" if len(ra) == len(rb) else f" (record counts {len(ra)} vs {len(rb)})"))

    ha, pa = read_dump(fa)
    hb, pb = read_dump(fb)
    a, b = pa[0], pb[0]
    ta = a['itype'].astype(int) if 'itype' in a else np.ones(len(a['x']), int)
    tb = b['itype'].astype(int) if 'itype' in b else np.ones(len(b['x']), int)
    xa = np.stack([a['x'], a['y'], a['z']], 1); xb = np.stack([b['x'], b['y'], b['z']], 1)
    va = np.stack([a['vx'], a['vy'], a['vz']], 1); vb = np.stack([b['vx'], b['vy'], b['vz']], 1)
    h_a = a['h'].astype(float); h_b = b['h'].astype(float)
    print(f"types    {'type':6}{'n':>10}{'dead A/B':>14}{'rms dv':>11}{'rms dx/h':>11}{'max dh/h':>11}")
    if (ta != tb).any():
        print("         particle types differ between the dumps"); ok = False
    for t in sorted(set(tb)):
        m = (tb == t)
        live = m & (h_a > 0) & (h_b > 0)
        dead = f"{int((m & (h_a <= 0)).sum())}/{int((m & (h_b <= 0)).sum())}"
        if not live.any():
            print(f"         {TYPENAMES.get(t, t):6}{int(m.sum()):10}{dead:>14}"); continue
        dv = np.sqrt(((va[live] - vb[live])**2).sum(1).mean()) / max(np.sqrt((vb[live]**2).sum(1).mean()), 1e-300)
        dx = np.sqrt(((xa[live] - xb[live])**2).sum(1).mean()) / h_b[live].mean()
        dh = np.abs(h_a[live] / h_b[live] - 1).max()
        print(f"         {TYPENAMES.get(t, t):6}{int(m.sum()):10}{dead:>14}{dv:11.2e}{dx:11.2e}{dh:11.2e}")

    na, ea = ev_table(A, prefix)
    nb, eb = ev_table(B, prefix)
    if ea is None or eb is None:
        print("ev       no .ev file in one of the runs")
    else:
        # the last time both files reached
        common = np.intersect1d(ea[:, 0], eb[:, 0])
        if len(common) == 0:
            print("ev       no common time")
        else:
            ia = np.where(ea[:, 0] == common[-1])[0][-1]; ib = np.where(eb[:, 0] == common[-1])[0][-1]
            # the conserved and bulk columns; centre-of-mass columns sit near zero, so a
            # relative figure for them says nothing
            scale = np.maximum(np.abs(eb).max(0), 1e-300)
            rel = np.abs(ea[ia] - eb[ib]) / scale
            keep = [j for j, n in enumerate(nb) if n in EVCOLS and j < ea.shape[1]]
            print(f"ev       t = {common[-1]:.6g} ({len(common)} common rows), |A-B|/max|B|: "
                  + ", ".join(f"{nb[j]} {rel[j]:.1e}" for j in keep))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
