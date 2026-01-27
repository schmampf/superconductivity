import numpy as np

from theory.models.functions import NDArray64

# Maximum number of sidebands (matches parameter ns=520 in iv.for)
NS_MAX: int = 520


def get_I_Delta(
    V_Delta: NDArray64,
    tau: float = 0.5,
    T_Delta: float = 0.0,
    Dynes_Delta: float = 1e-6,
    w_min: float = -15.0,
    w_max: float = 15.0,
) -> NDArray64:
    """
    Compute the dc current I(V) for a single-channel superconducting point contact,
    based on the Fortran iv.for code by Cuevas et al.

    Parameters
    ----------
    V_Delta :
        1D array of voltages in units of Delta (eV/Delta), e.g. from -4 to 4.
    tau :
        Normal-state transmission of the single channel (0 < tau <= 1).
    T_Delta :
        Temperature in units of Delta (k_B T / Delta). If 0.0, a tiny value
        is used (1e-7) as in the Fortran code to avoid tanh(∞) issues.
    Dynes_Delta :
        Dynes broadening parameter eta in units of Delta.
    w_min, w_max :
        Energy integration limits in units of Delta. Fortran reads these from
        iv.in; here we expose them as parameters with sensible defaults.

    Returns
    -------
    I_Delta :
        1D array of dc currents (dimensionless units, proportional to e Delta / h).
        You can later scale to physical units if desired.
    """
    V_Delta = np.asarray(V_Delta, dtype=np.float64)
    tau = float(tau)
    temp = float(T_Delta)
    eta = float(Dynes_Delta)

    # Fortran: if (temp.eq.0.0) temp = 1.e-7
    if temp == 0.0:
        temp = 1e-7

    # Fortran: thop = sqrt((2.0-trans-2.0*sqrt(1.0-trans))/trans)
    thop = np.sqrt((2.0 - tau - 2.0 * np.sqrt(1.0 - tau)) / tau)

    I_Delta = np.empty_like(V_Delta, dtype=np.float64)

    for idx, v in enumerate(V_Delta):
        if v == 0.0:
            # Avoid division by zero in nan = int(2.0/abs(v)); current is finite.
            I_Delta[idx] = 0.0
            continue

        # Fortran:
        #   nan = int(2.0/abs(v))
        #   if (mod(nan,2).eq.0) then
        #       nan = nan + 7
        #   else
        #       nan = nan + 6
        #   end if
        nan = int(2.0 / abs(v))
        if nan % 2 == 0:
            nan += 7
        else:
            nan += 6

        # Enforce the maximum number of MAR sidebands (ns)
        if nan > NS_MAX:
            nan = NS_MAX

        # Fortran: current = real(zint(wi,wf,Atol,Rtol,ierr))
        curri = _zint(
            w_min=w_min,
            w_max=w_max,
            v=v,
            temp=temp,
            thop=thop,
            nan=nan,
            eta=eta,
        )
        I_Delta[idx] = curri.real

    return I_Delta


def _inv2x2(a: np.ndarray) -> np.ndarray:
    """
    Invert a 2x2 complex matrix using the same formula as the Fortran `inv` subroutine.
    """
    det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    ainv = np.empty_like(a)
    ainv[0, 0] = a[1, 1] / det
    ainv[0, 1] = -a[0, 1] / det
    ainv[1, 0] = -a[1, 0] / det
    ainv[1, 1] = a[0, 0] / det
    return ainv


def _zint(
    w_min: float,
    w_max: float,
    v: float,
    temp: float,
    thop: float,
    nan: int,
    eta: float,
) -> complex:
    """
    Adaptive complex Simpson integration of _zintegrand(w), translated from
    the Fortran zint/zintrp implementation in iv.for.

    Parameters
    ----------
    w_min, w_max :
        Integration limits.
    v, temp, thop, nan, eta :
        Parameters forwarded to _zintegrand.

    Returns
    -------
    zint_val :
        Complex integral of _zintegrand(w) from w_min to w_max.
    """
    # Fortran tolerances: Atol = (1e-8, 1.0), Rtol = (1e-6, 1.0)
    Atol = 1e-8 + 1.0j
    Rtol = 1e-6 + 1.0j

    # Maximum stack size (Fortran: stksz = 4096)
    stksz = 4096

    # Stacks for x and z values; we store triples (lx, mx, ux) and (lz, mz, uz)
    xstk: list[float] = []
    zstk: list[complex] = []

    istk = 0
    ierr = 0  # not used further, but kept for diagnostics / parity

    # Initial three points for Simpson integral
    lx = float(w_min)
    ux = float(w_max)
    mx = 0.5 * (lx + ux)

    lz = _zintegrand(w=lx, v=v, temp=temp, thop=thop, nan=nan, eta=eta)
    mz = _zintegrand(w=mx, v=v, temp=temp, thop=thop, nan=nan, eta=eta)
    uz = _zintegrand(w=ux, v=v, temp=temp, thop=thop, nan=nan, eta=eta)

    zint_val: complex = 0.0 + 0.0j

    while True:
        # Perform local refinement over current [lx, ux]
        (
            zres,
            itask,
            lx,
            lz,
            mx,
            mz,
            ux,
            uz,
            xstk,
            zstk,
            istk,
        ) = _zintrp(
            lx=lx,
            lz=lz,
            mx=mx,
            mz=mz,
            ux=ux,
            uz=uz,
            xstk=xstk,
            zstk=zstk,
            istk=istk,
            v=v,
            temp=temp,
            thop=thop,
            nan=nan,
            eta=eta,
            Atol=Atol,
            Rtol=Rtol,
        )

        # Stack size check (Fortran: if 3*(istk+1) .ge. stksz then error)
        if 3 * (istk + 1) >= stksz:
            raise RuntimeError("zint: stack size exceeded (stksz=4096)")

        # Track maximum stack depth as in ierr
        if istk > ierr:
            ierr = istk

        # If the current branch has converged, accumulate the result
        if itask == 0:
            zint_val += zres
        else:
            # Not converged: loop again on the updated lower-half segment
            continue

        # If the stack is not empty, pop the next segment and integrate it
        if istk != 0:
            istk -= 1

            # Pop last triple from stacks
            ux = xstk.pop()
            mx = xstk.pop()
            lx = xstk.pop()

            uz = zstk.pop()
            mz = zstk.pop()
            lz = zstk.pop()

            # Continue with this new segment
            continue

        # Stack empty and last branch converged: integral is complete
        break

    return zint_val


def _zintrp(
    lx: float,
    lz: complex,
    mx: float,
    mz: complex,
    ux: float,
    uz: complex,
    xstk: list[float],
    zstk: list[complex],
    istk: int,
    v: float,
    temp: float,
    thop: float,
    nan: int,
    eta: float,
    Atol: complex,
    Rtol: complex,
) -> tuple[
    complex,
    int,
    float,
    complex,
    float,
    complex,
    float,
    complex,
    list[float],
    list[complex],
    int,
]:
    """
    Recursive part of the adaptive Simpson integrator (Fortran zintrp).

    Returns
    -------
    zres :
        Refined integral estimate over [lx, ux].
    itask :
        0 if converged on this segment, 1 if further subdivision is needed.
    lx, lz, mx, mz, ux, uz :
        Possibly updated segment endpoints and function values (lower half if split).
    xstk, zstk, istk :
        Updated stacks and stack index.
    """
    two = 2.0
    three = 3.0
    four = 4.0

    # Three-point Simpson rule (coarse estimate)
    dx = (ux - lx) / two
    ctest = dx * (lz + four * mz + uz) / three

    # Five-point Simpson rule (refined estimate)
    dx = dx / two
    lmx = 0.5 * (lx + mx)
    lmz = _zintegrand(w=lmx, v=v, temp=temp, thop=thop, nan=nan, eta=eta)
    umx = 0.5 * (ux + mx)
    umz = _zintegrand(w=umx, v=v, temp=temp, thop=thop, nan=nan, eta=eta)
    zres = dx * (lz + two * mz + four * (lmz + umz) + uz) / three

    diff = zres - ctest
    summ = zres + ctest

    # Helper to avoid div-by-zero in relative tests
    def _safe_div(num: float, den: float) -> float:
        if den == 0.0:
            return float("inf")
        return num / den

    # Real-part tolerance checks (matches Fortran logic)
    if abs(diff.real) < Atol.real:
        # Imag part absolute tolerance
        if abs(diff.imag) < Atol.imag:
            itask = 0
        else:
            tmpi = two * abs(diff.imag)
            tmpi = _safe_div(tmpi, abs(summ.imag))
            itask = 0 if tmpi < Rtol.imag else 1
    else:
        # Real part relative tolerance
        tmpr = two * abs(diff.real)
        tmpr = _safe_div(tmpr, abs(summ.real))
        if tmpr < Rtol.real:
            # Now check imaginary part
            if abs(diff.imag) < Atol.imag:
                itask = 0
            else:
                tmpi = two * abs(diff.imag)
                tmpi = _safe_div(tmpi, abs(summ.imag))
                itask = 0 if tmpi < Rtol.imag else 1
        else:
            itask = 1

    # If not converged, push upper half on stack and continue with lower half
    if itask == 1:
        # Push upper segment [mx, ux] with midpoint umx
        xstk.extend([mx, umx, ux])
        zstk.extend([mz, umz, uz])
        istk += 1

        # Update current segment to lower half [lx, mx] with midpoint lmx
        ux = mx
        mx = lmx
        uz = mz
        mz = lmz

    return zres, itask, lx, lz, mx, mz, ux, uz, xstk, zstk, istk


def _zintegrand(
    w: float,
    v: float,
    temp: float,
    thop: float,
    nan: int,
    eta: float,
) -> complex:
    """
    Complex current density zintegrand(w).

    This currently implements:
    - the 'surface Green functions' block
    - the er/ea, vpr/vmr, vpa/vma block

    The self-energies, T-matrix recursion and current density will follow
    in later steps.
    """
    ui = 1j
    delta = 1.0
    t2 = thop**2  # Fortran: t2 = thop**2.0

    # Pauli matrix tau3 in Nambu space
    tau3 = np.zeros((2, 2), dtype=np.float64)
    tau3[0, 0] = 1.0
    tau3[1, 1] = -1.0

    # j runs from -nan-2 to nan+2 in Fortran; map to 0..n_j-1 in Python
    j_min = -nan - 2
    j_max = nan + 2
    n_j = j_max - j_min + 1

    # Allocate Green's functions: shape (2, 2, n_j)
    g0lr = np.zeros((2, 2, n_j), dtype=np.complex128)
    g0rr = np.zeros((2, 2, n_j), dtype=np.complex128)
    g0la = np.zeros((2, 2, n_j), dtype=np.complex128)
    g0ra = np.zeros((2, 2, n_j), dtype=np.complex128)
    g0kl = np.zeros((2, 2, n_j), dtype=np.complex128)
    g0kr = np.zeros((2, 2, n_j), dtype=np.complex128)

    # NEW: allocate er/ea, vpr/vmr, vpa/vma on the same j-grid
    er = np.zeros((2, 2, n_j), dtype=np.complex128)
    ea = np.zeros((2, 2, n_j), dtype=np.complex128)
    vpr = np.zeros((2, 2, n_j), dtype=np.complex128)
    vmr = np.zeros((2, 2, n_j), dtype=np.complex128)
    vpa = np.zeros((2, 2, n_j), dtype=np.complex128)
    vma = np.zeros((2, 2, n_j), dtype=np.complex128)

    # --- Surface Green functions loop (Fortran: do j=-nan-2,nan+2) ---
    for j in range(j_min, j_max + 1):
        idx_j = j - j_min  # 0-based index into our arrays

        # wj = w + float(j)*v
        wj = w + float(j) * v
        # wwj = wj + ui*eta
        wwj = wj + ui * eta
        # omega = csqrt(delta**2 - wwj**2)
        omega = np.sqrt(delta**2 - wwj**2)

        # Left retarded Green function g0lr (2x2 Nambu matrix)
        g0lr[0, 0, idx_j] = -wwj / omega
        g0lr[0, 1, idx_j] = delta / omega
        g0lr[1, 0, idx_j] = -delta / omega
        g0lr[1, 1, idx_j] = wwj / omega

        # Left advanced: g0la = (g0lr)^\dagger with sign convention
        g0la[0, 0, idx_j] = np.conjugate(g0lr[0, 0, idx_j])
        g0la[0, 1, idx_j] = -np.conjugate(g0lr[1, 0, idx_j])
        g0la[1, 0, idx_j] = -np.conjugate(g0lr[0, 1, idx_j])
        g0la[1, 1, idx_j] = np.conjugate(g0lr[1, 1, idx_j])

        # Right retarded = left retarded (symmetric junction)
        g0rr[:, :, idx_j] = g0lr[:, :, idx_j]

        # Right advanced: g0ra = (g0rr)^\dagger
        g0ra[0, 0, idx_j] = np.conjugate(g0rr[0, 0, idx_j])
        g0ra[0, 1, idx_j] = -np.conjugate(g0rr[1, 0, idx_j])
        g0ra[1, 0, idx_j] = -np.conjugate(g0rr[0, 1, idx_j])
        g0ra[1, 1, idx_j] = np.conjugate(g0rr[1, 1, idx_j])

        # Keldysh components: g0k = (g0r - g0a) * tanh(wj / 2T)
        fact = np.tanh(0.5 * wj / temp)
        g0kl[:, :, idx_j] = (g0lr[:, :, idx_j] - g0la[:, :, idx_j]) * fact
        g0kr[:, :, idx_j] = (g0rr[:, :, idx_j] - g0ra[:, :, idx_j]) * fact

    # --- Parameters of the algebraic system for T-matrix components ---
    # Fortran: DO j=-nan,nan,2
    for j in range(-nan, nan + 1, 2):
        idx_j = j - j_min
        idx_jp1 = (j + 1) - j_min
        idx_jm1 = (j - 1) - j_min
        idx_jp2 = (j + 2) - j_min
        idx_jm2 = (j - 2) - j_min

        # er (retarded)
        er[0, 0, idx_j] = 1.0 - t2 * g0rr[0, 0, idx_jp1] * g0lr[0, 0, idx_j]
        er[0, 1, idx_j] = -t2 * g0rr[0, 0, idx_jp1] * g0lr[0, 1, idx_j]
        er[1, 0, idx_j] = -t2 * g0rr[1, 1, idx_jm1] * g0lr[1, 0, idx_j]
        er[1, 1, idx_j] = 1.0 - t2 * g0rr[1, 1, idx_jm1] * g0lr[1, 1, idx_j]

        # ea (advanced)
        ea[0, 0, idx_j] = 1.0 - t2 * g0ra[0, 0, idx_jp1] * g0la[0, 0, idx_j]
        ea[0, 1, idx_j] = -t2 * g0ra[0, 0, idx_jp1] * g0la[0, 1, idx_j]
        ea[1, 0, idx_j] = -t2 * g0ra[1, 1, idx_jm1] * g0la[1, 0, idx_j]
        ea[1, 1, idx_j] = 1.0 - t2 * g0ra[1, 1, idx_jm1] * g0la[1, 1, idx_j]

        # vpr (retarded "plus" coupling)
        vpr[0, 0, idx_j] = t2 * g0rr[0, 1, idx_jp1] * g0lr[1, 0, idx_jp2]
        vpr[0, 1, idx_j] = t2 * g0rr[0, 1, idx_jp1] * g0lr[1, 1, idx_jp2]
        vpr[1, 0, idx_j] = 0.0 + 0.0j
        vpr[1, 1, idx_j] = 0.0 + 0.0j

        # vpa (advanced "plus" coupling)
        vpa[0, 0, idx_j] = t2 * g0ra[0, 1, idx_jp1] * g0la[1, 0, idx_jp2]
        vpa[0, 1, idx_j] = t2 * g0ra[0, 1, idx_jp1] * g0la[1, 1, idx_jp2]
        vpa[1, 0, idx_j] = 0.0 + 0.0j
        vpa[1, 1, idx_j] = 0.0 + 0.0j

        # vmr (retarded "minus" coupling)
        vmr[0, 0, idx_j] = 0.0 + 0.0j
        vmr[0, 1, idx_j] = 0.0 + 0.0j
        vmr[1, 0, idx_j] = t2 * g0rr[1, 0, idx_jm1] * g0lr[0, 0, idx_jm2]
        vmr[1, 1, idx_j] = t2 * g0rr[1, 0, idx_jm1] * g0lr[0, 1, idx_jm2]

        # vma (advanced "minus" coupling)
        vma[0, 0, idx_j] = 0.0 + 0.0j
        vma[0, 1, idx_j] = 0.0 + 0.0j
        vma[1, 0, idx_j] = t2 * g0ra[1, 0, idx_jm1] * g0la[0, 0, idx_jm2]
        vma[1, 1, idx_j] = t2 * g0ra[1, 0, idx_jm1] * g0la[0, 1, idx_jm2]

    # --- SELF-ENERGIES: adr, air, ada, aia (Fortran "SELF-ENERGIES" block) ---

    adr = np.zeros((2, 2, n_j), dtype=np.complex128)
    air = np.zeros((2, 2, n_j), dtype=np.complex128)
    ada = np.zeros((2, 2, n_j), dtype=np.complex128)
    aia = np.zeros((2, 2, n_j), dtype=np.complex128)

    # boundary at j = ±nan
    idx_nan = nan - j_min  # index corresponding to j = +nan
    idx_mnan = (-nan) - j_min  # index corresponding to j = -nan

    # aux1r/aux3r/aux1a/aux3a at the boundaries
    aux1r = er[:, :, idx_nan].copy()
    aux3r = er[:, :, idx_mnan].copy()
    aux1a = ea[:, :, idx_nan].copy()
    aux3a = ea[:, :, idx_mnan].copy()

    # invert (Fortran: call inv(...))
    aux2r = _inv2x2(aux1r)
    aux4r = _inv2x2(aux3r)
    aux2a = _inv2x2(aux1a)
    aux4a = _inv2x2(aux3a)

    # store into adr/air/ada/aia at j = ±nan
    adr[:, :, idx_nan] = aux2r
    air[:, :, idx_mnan] = aux4r
    ada[:, :, idx_nan] = aux2a
    aia[:, :, idx_mnan] = aux4a

    # recursive sweep: Fortran "DO i=nan-2,1,-2"
    for i in range(nan - 2, 0, -2):  # i = nan-2, nan-4, ..., 1
        idx_i = i - j_min
        idx_ip2 = (i + 2) - j_min
        idx_mi = (-i) - j_min
        idx_mip2 = (-i - 2) - j_min

        # start from er/ea at j = ±i
        aux1r = er[:, :, idx_i].copy()
        aux3r = er[:, :, idx_mi].copy()
        aux1a = ea[:, :, idx_i].copy()
        aux3a = ea[:, :, idx_mi].copy()

        # nested sums over k,l,i1,i2
        for k in range(2):
            for l in range(2):
                for i1 in range(2):
                    for i2 in range(2):
                        # retarded
                        aux1r[k, l] -= (
                            vpr[k, i1, idx_i]
                            * adr[i1, i2, idx_ip2]
                            * vmr[i2, l, idx_ip2]
                        )
                        aux3r[k, l] -= (
                            vmr[k, i1, idx_mi]
                            * air[i1, i2, idx_mip2]
                            * vpr[i2, l, idx_mip2]
                        )

                        # advanced
                        aux1a[k, l] -= (
                            vpa[k, i1, idx_i]
                            * ada[i1, i2, idx_ip2]
                            * vma[i2, l, idx_ip2]
                        )
                        aux3a[k, l] -= (
                            vma[k, i1, idx_mi]
                            * aia[i1, i2, idx_mip2]
                            * vpa[i2, l, idx_mip2]
                        )

        # invert to get adr/air/ada/aia at ±i
        aux2r = _inv2x2(aux1r)
        aux4r = _inv2x2(aux3r)
        aux2a = _inv2x2(aux1a)
        aux4a = _inv2x2(aux3a)

        adr[:, :, idx_i] = aux2r
        air[:, :, idx_mi] = aux4r
        ada[:, :, idx_i] = aux2a
        aia[:, :, idx_mi] = aux4a
        # --- Closed system for T_{1,0} and T_{-1,0} ---

    # Allocate full T-matrix arrays on the same j-grid as the Green functions
    tr = np.zeros((2, 2, n_j), dtype=np.complex128)
    ta = np.zeros((2, 2, n_j), dtype=np.complex128)

    # Helper indices for j = +1, +3, -1
    idx_p1 = 1 - j_min  # j = +1
    idx_p3 = 3 - j_min  # j = +3
    idx_m1 = -1 - j_min  # j = -1

    # tx --> v_{1,0}  and ty --> v_{-1,0}
    tx = np.zeros((2, 2), dtype=np.complex128)
    ty = np.zeros((2, 2), dtype=np.complex128)
    tx[1, 1] = thop  # Fortran: tx(2,2) = thop
    ty[0, 0] = thop  # Fortran: ty(1,1) = thop

    # aux/cp matrices for the closed system
    aux1r = np.zeros((2, 2), dtype=np.complex128)
    aux1a = np.zeros((2, 2), dtype=np.complex128)
    cpr = np.zeros((2, 2), dtype=np.complex128)
    cpa = np.zeros((2, 2), dtype=np.complex128)

    # Fortran:
    # do k=1,2
    #   do l=1,2
    #      aux1r(k,l) = er(k,l,1)
    #      aux1a(k,l) = ea(k,l,1)
    #      cpr(k,l) = tx(k,l)
    #      cpa(k,l) = tx(k,l)
    #      ...
    for k in range(2):
        for l in range(2):
            aux1r[k, l] = er[k, l, idx_p1]
            aux1a[k, l] = ea[k, l, idx_p1]
            cpr[k, l] = tx[k, l]
            cpa[k, l] = tx[k, l]
            for i1 in range(2):
                for i2 in range(2):
                    # retarded part
                    aux1r[k, l] -= (
                        vpr[k, i1, idx_p1] * adr[i1, i2, idx_p3] * vmr[i2, l, idx_p3]
                    )
                    aux1r[k, l] -= (
                        vmr[k, i1, idx_p1] * air[i1, i2, idx_m1] * vpr[i2, l, idx_m1]
                    )

                    # advanced part
                    aux1a[k, l] -= (
                        vpa[k, i1, idx_p1] * ada[i1, i2, idx_p3] * vma[i2, l, idx_p3]
                    )
                    aux1a[k, l] -= (
                        vma[k, i1, idx_p1] * aia[i1, i2, idx_m1] * vpa[i2, l, idx_m1]
                    )

                    # inhomogeneous terms (cpr, cpa)
                    cpr[k, l] += vmr[k, i1, idx_p1] * air[i1, i2, idx_m1] * ty[i2, l]
                    cpa[k, l] += vma[k, i1, idx_p1] * aia[i1, i2, idx_m1] * ty[i2, l]

    # Solve the 2x2 systems for j = +1
    aux2r = _inv2x2(aux1r)
    aux2a = _inv2x2(aux1a)

    # Fortran:
    # tr(i,j,1) = sum_i1 aux2r(i,i1) * cpr(i1,j)
    # ta(i,j,1) = sum_i1 aux2a(i,i1) * cpa(i1,j)
    for i in range(2):
        for j in range(2):
            val_tr = 0.0 + 0.0j
            val_ta = 0.0 + 0.0j
            for i1 in range(2):
                val_tr += aux2r[i, i1] * cpr[i1, j]
                val_ta += aux2a[i, i1] * cpa[i1, j]
            tr[i, j, idx_p1] = val_tr
            ta[i, j, idx_p1] = val_ta

    # Now obtain j = -1 using cmr/cma
    cmr = ty.copy()
    cma = ty.copy()
    for k in range(2):
        for l in range(2):
            for i1 in range(2):
                cmr[k, l] += vpr[k, i1, idx_m1] * tr[i1, l, idx_p1]
                cma[k, l] += vpa[k, i1, idx_m1] * ta[i1, l, idx_p1]

    # Fortran:
    # tr(i,j,-1) = sum_i1 air(i,i1,-1) * cmr(i1,j)
    # ta(i,j,-1) = sum_i1 aia(i,i1,-1) * cma(i1,j)
    for i in range(2):
        for j in range(2):
            val_tr = 0.0 + 0.0j
            val_ta = 0.0 + 0.0j
            for i1 in range(2):
                val_tr += air[i, i1, idx_m1] * cmr[i1, j]
                val_ta += aia[i, i1, idx_m1] * cma[i1, j]
            tr[i, j, idx_m1] = val_tr
            ta[i, j, idx_m1] = val_ta

    # --- Recursive T-matrix relations for |j| >= 3 (Fortran: DO j=3,nan,2) ---
    for j in range(3, nan + 1, 2):
        idx_j = j - j_min
        idx_jm2 = (j - 2) - j_min
        idx_mj = (-j) - j_min
        idx_mjp2 = (-j + 2) - j_min

        for k in range(2):
            for l in range(2):
                # initialize T-matrices at ±j
                tr[k, l, idx_j] = 0.0 + 0.0j
                ta[k, l, idx_j] = 0.0 + 0.0j
                tr[k, l, idx_mj] = 0.0 + 0.0j
                ta[k, l, idx_mj] = 0.0 + 0.0j

                for i1 in range(2):
                    for i2 in range(2):
                        # positive j branch
                        tr[k, l, idx_j] += (
                            adr[k, i1, idx_j] * vmr[i1, i2, idx_j] * tr[i2, l, idx_jm2]
                        )
                        ta[k, l, idx_j] += (
                            ada[k, i1, idx_j] * vma[i1, i2, idx_j] * ta[i2, l, idx_jm2]
                        )

                        # negative j branch
                        tr[k, l, idx_mj] += (
                            air[k, i1, idx_mj]
                            * vpr[i1, i2, idx_mj]
                            * tr[i2, l, idx_mjp2]
                        )
                        ta[k, l, idx_mj] += (
                            aia[k, i1, idx_mj]
                            * vpa[i1, i2, idx_mj]
                            * ta[i2, l, idx_mjp2]
                        )

    # --- Current density (Fortran "Current density" block) ---
    curr = 0.0
    idx_0 = 0 - j_min  # index corresponding to j = 0

    for j in range(-nan, nan + 1, 2):
        idx_j = j - j_min
        for i1 in range(2):
            t31 = tau3[i1, i1]
            for i2 in range(2):
                for i3 in range(2):
                    t33 = tau3[i3, i3]
                    for i4 in range(2):
                        t34 = tau3[i4, i4]

                        term1 = (
                            t31
                            * tr[i1, i2, idx_j]
                            * g0kr[i2, i3, idx_0]
                            * t33
                            * np.conjugate(tr[i4, i3, idx_j])
                            * t34
                            * g0la[i4, i1, idx_j]
                        )

                        term2 = (
                            t31
                            * g0rr[i1, i2, idx_0]
                            * tau3[i2, i2]
                            * np.conjugate(ta[i3, i2, idx_j])
                            * t33
                            * g0kl[i3, i4, idx_j]
                            * ta[i4, i1, idx_j]
                        )

                        curr += (term1 + term2).real

    curr *= 0.5
    return curr + 0.0j
