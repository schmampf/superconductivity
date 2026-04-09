module ha_sym_api
  implicit none
  integer, parameter :: ns = 520

  ! Mirror the COMMON blocks from ha_sym.for
  real :: v, temp, thop
  real :: eta
  integer :: nan

  common /argum/ v, temp, thop
  common /dynes/ eta
  common /dim/  nan

contains

  subroutine ha_prepare_state(trans_in, temp_in, eta_in, v_in)
    implicit none
    real, intent(in) :: trans_in, temp_in, eta_in, v_in
    real :: trans_local

    temp = temp_in
    if (temp == 0.0) temp = 1.e-7

    eta = eta_in
    if (eta <= 0.0) eta = 1.e-6

    trans_local = trans_in
    if (trans_local <= 0.0) then
       thop = 0.0
    else
       thop = sqrt((2.0-trans_local-2.0*sqrt(1.0-trans_local))/trans_local)
    end if

    v = v_in

    if (abs(v) < 1.e-12) then
       nan = ns
    else
       nan = int(2.0/abs(v))
       if (mod(nan,2) == 0) then
          nan = nan + 7
       else
          nan = nan + 6
       end if
    end if
    if (nan > ns) nan = ns
  end subroutine ha_prepare_state

  function ha_sym_curve(trans_in, temp_in, eta_in, wi, wf, voltages) result(currents)
    implicit none
    real, intent(in) :: trans_in, temp_in, eta_in, wi, wf
    real, intent(in) :: voltages(:)
    real :: currents(size(voltages))

    complex :: Atol, Rtol
    complex, external :: zint
    integer :: ierr, k

    Atol = (1.e-8, 1.0)
    Rtol = (1.e-6, 1.0)
    ierr = 0

    do k = 1, size(voltages)
       call ha_prepare_state(trans_in, temp_in, eta_in, voltages(k))
       currents(k) = real(zint(wi, wf, Atol, Rtol, ierr))
    end do
  end function ha_sym_curve

end module ha_sym_api
