module fcs_api
  implicit none

contains

  subroutine fcs_prepare_curve_state(trans_in, temp_in, dmvt1_in, dmvt2_in, eta1_in, eta2_in, nmax_in, nchi_in, &
       trans_out, temp_out, dmvt1_out, dmvt2_out, eta1_out, eta2_out, nmax_out, nchi_out)
    implicit none
    real*8, intent(in) :: trans_in, temp_in, dmvt1_in, dmvt2_in
    real*8, intent(in) :: eta1_in, eta2_in
    integer, intent(in) :: nmax_in, nchi_in
    real*8, intent(out) :: trans_out, temp_out, dmvt1_out, dmvt2_out
    real*8, intent(out) :: eta1_out, eta2_out
    integer, intent(out) :: nmax_out, nchi_out

    trans_out = trans_in
    temp_out = temp_in
    if (temp_out .eq. 0.d0) temp_out = 1.d-7

    eta1_out = eta1_in
    eta2_out = eta2_in
    if (eta1_out .eq. 0.d0) eta1_out = 1.d-6
    if (eta2_out .eq. 0.d0) eta2_out = 1.d-6

    dmvt1_out = dmvt1_in
    dmvt2_out = dmvt2_in
    nmax_out = nmax_in
    nchi_out = nchi_in
  end subroutine fcs_prepare_curve_state

  subroutine fcs_prepare_voltage_state(v_in, v_out)
    implicit none
    real*8, intent(in) :: v_in
    real*8, intent(out) :: v_out

    v_out = v_in
  end subroutine fcs_prepare_voltage_state

  function fcs_curve(trans_in, temp_in, dmvt1_in, dmvt2_in, eta1_in, eta2_in, voltages, nmax_in, iw_in, nchi_in) result(currents)
    implicit none
    real*8, intent(in) :: trans_in, temp_in, dmvt1_in, dmvt2_in
    real*8, intent(in) :: eta1_in, eta2_in
    real*8, intent(in) :: voltages(:)
    integer, intent(in) :: nmax_in, iw_in, nchi_in
    real*8 :: currents(size(voltages), nmax_in + 1)

    integer, parameter :: ns = 2000
    complex*16 :: cvec(3)
    complex*16, allocatable :: p(:)
    real*8, allocatable :: curr(:), f(:)
    complex*16, allocatable :: g1r(:), f1r(:), g1a(:), f1a(:)
    complex*16, allocatable :: g2r(:), f2r(:), g2a(:), f2a(:)
    complex*16, allocatable :: a(:, :, :), b(:, :, :), c(:, :, :)
    complex*16, allocatable :: aux1(:, :), aux2(:, :), work(:)
    integer, allocatable :: ipiv(:)
    real*8 :: current, snoise, c3
    real*8 :: wi, wf, de, dee, w, v_eval
    integer :: i, j, k
    real*8 :: trans_eval, temp_eval, dmvt1_eval, dmvt2_eval
    real*8 :: eta1_eval, eta2_eval
    integer :: nmax_eval, nchi_eval

    if (size(voltages) .eq. 0) return

    call fcs_prepare_curve_state( &
         trans_in, temp_in, dmvt1_in, dmvt2_in, eta1_in, eta2_in, nmax_in, nchi_in, &
         trans_eval, temp_eval, dmvt1_eval, dmvt2_eval, eta1_eval, eta2_eval, nmax_eval, nchi_eval &
    )
    allocate( &
         p(-ns:ns), curr(-ns:ns), f(-ns:ns), &
         g1r(-ns:ns), f1r(-ns:ns), g1a(-ns:ns), f1a(-ns:ns), &
         g2r(-ns:ns), f2r(-ns:ns), g2a(-ns:ns), f2a(-ns:ns), &
         a(4, 4, -ns:ns), b(4, 4, -ns:ns), c(4, 4, -ns:ns), &
         aux1(4, 4), aux2(4, 4), work(4**2), ipiv(4) &
    )

    do k = 1, size(voltages)
       call fcs_prepare_voltage_state(voltages(k), v_eval)
       wi = -v_eval
       wf = 0.d0

       if (iw_in .le. 0) then
          currents(k, :) = 0.d0
       else
          de = (wf - wi) / dfloat(iw_in)
          current = 0.d0
          snoise = 0.d0
          c3 = 0.d0
          do j = -2*nmax_eval-2, 2*nmax_eval+2
             curr(j) = 0.d0
          end do

          do i = 0, iw_in
             w = wi + dfloat(i) * de
             dee = de
             call green(cvec, w, 3, p, v_eval, trans_eval, temp_eval, &
                  dmvt1_eval, dmvt2_eval, eta1_eval, eta2_eval, &
                  nmax_eval, nchi_eval, &
                  f, g1r, f1r, g1a, f1a, g2r, f2r, g2a, f2a, &
                  a, b, c, aux1, aux2, work, ipiv)
             if (i .eq. 0 .or. i .eq. iw_in) dee = de / 2.d0
             current = current + dreal(cvec(1)) * dee
             snoise = snoise + dreal(cvec(2)) * dee
             c3 = c3 + dreal(cvec(3)) * dee
             do j = -2*nmax_eval-2, 2*nmax_eval+2
                curr(j) = curr(j) + dabs(dfloat(j)) * dreal(p(j)) * dee
             end do
          end do

          currents(k, 1) = current * 77.4809d0
          do j = 1, nmax_in
             currents(k, j+1) = (curr(-j) - curr(j)) * 77.4809d0
          end do
       end if
    end do
    deallocate( &
         p, curr, f, g1r, f1r, g1a, f1a, &
         g2r, f2r, g2a, f2a, a, b, c, &
         aux1, aux2, work, ipiv &
    )
  end function fcs_curve

end module fcs_api
