module ha_asym_api
  implicit none

contains

  ! Prepare the reduced kernel state. Temperature-dependent gaps are
  ! computed in Python before calling into this backend.
  subroutine ha_prepare_curve_state(trans_in, temp_in, dmvt1_in, dmvt2_in, eta1_in, eta2_in, &
       dmvt1_out, dmvt2_out, eta1_out, eta2_out, t_out, t2_out, temp_out, mmax_out)
    implicit none
    real*8, intent(in) :: trans_in, temp_in, dmvt1_in, dmvt2_in
    real*8, intent(in) :: eta1_in, eta2_in
    real*8, intent(out) :: dmvt1_out, dmvt2_out, eta1_out, eta2_out
    real*8, intent(out) :: t_out, t2_out, temp_out
    integer, intent(out) :: mmax_out

    dmvt1_out = dmvt1_in
    dmvt2_out = dmvt2_in

    eta1_out = eta1_in
    eta2_out = eta2_in
    if (eta1_out .le. 0.d0) eta1_out = 1.d-6
    if (eta2_out .le. 0.d0) eta2_out = 1.d-6

    temp_out = temp_in
    if (temp_out .eq. 0.d0) temp_out = 1.d-7

    if (trans_in .le. 0.d0) then
       t_out = 0.d0
    else
       t_out = dsqrt((2.d0 - trans_in - 2.d0 * dsqrt(1.d0 - trans_in)) / trans_in)
    end if
    t2_out = t_out * t_out
    mmax_out = 0
  end subroutine ha_prepare_curve_state

  subroutine ha_prepare_voltage_state(v_in, v_out, n_out)
    implicit none
    real*8, intent(in) :: v_in
    real*8, intent(out) :: v_out
    integer, intent(out) :: n_out

    v_out = v_in
    if (dabs(v_out) .lt. 1.d-12) then
       n_out = 1
    else
       n_out = int(2.d0 / dabs(v_out))
       if (mod(n_out, 2) .eq. 0) then
          n_out = n_out + 6
       else
          n_out = n_out + 7
       end if
    end if
  end subroutine ha_prepare_voltage_state

  function ha_asym_curve(trans_in, temp_in, dmvt1_in, dmvt2_in, eta1_in, eta2_in, voltages) result(currents)
    implicit none
    real*8, intent(in) :: trans_in, temp_in, dmvt1_in, dmvt2_in
    real*8, intent(in) :: eta1_in, eta2_in
    real*8, intent(in) :: voltages(:)
    real*8 :: currents(size(voltages))

    complex*16 :: Atol, Rtol
    complex*16, external :: zint
    integer :: ierr, k
    real*8 :: wi, wf
    real*8 :: dmvt1, dmvt2, eta1, eta2
    real*8 :: t, t2, temp, v
    integer :: n, m
    common/param/dmvt1,dmvt2,eta1,eta2
    common/dim/n,m
    common/hop/t,t2,temp,v

    if (size(voltages) .eq. 0) return

    call ha_prepare_curve_state( &
         trans_in, temp_in, dmvt1_in, dmvt2_in, eta1_in, eta2_in, &
         dmvt1, dmvt2, eta1, eta2, t, t2, temp, m &
    )

    Atol = (1.d-12, 1.d-12)
    Rtol = (1.d-9, 1.d-9)
    ierr = 0
    wf = maxval(dabs(voltages)) * 5.d0
    wi = -wf

    do k = 1, size(voltages)
       call ha_prepare_voltage_state(voltages(k), v, n)
       if (dabs(v) .lt. 0.002d0) then
          currents(k) = 0.d0
       else
          currents(k) = dreal(zint(wi, wf, Atol, Rtol, ierr))
       end if
    end do
  end function ha_asym_curve

end module ha_asym_api
