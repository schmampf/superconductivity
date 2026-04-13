module ha_sym_api
  implicit none
  integer, parameter :: ns = 520

contains

  subroutine ha_prepare_state(trans_in, temp_in, eta_in, v_in, temp_out, thop_out, eta_out, nan_out)
    implicit none
    real, intent(in) :: trans_in, temp_in, eta_in, v_in
    real, intent(out) :: temp_out, thop_out, eta_out
    integer, intent(out) :: nan_out
    real :: trans_local

    temp_out = temp_in
    if (temp_out == 0.0) temp_out = 1.e-7

    eta_out = eta_in
    if (eta_out <= 0.0) eta_out = 1.e-6

    trans_local = trans_in
    if (trans_local <= 0.0) then
       thop_out = 0.0
    else
       thop_out = sqrt((2.0-trans_local-2.0*sqrt(1.0-trans_local))/trans_local)
    end if

    if (abs(v_in) < 1.e-12) then
       nan_out = ns
    else
       nan_out = int(2.0/abs(v_in))
       if (mod(nan_out,2) == 0) then
          nan_out = nan_out + 7
       else
          nan_out = nan_out + 6
       end if
    end if
    if (nan_out > ns) nan_out = ns
  end subroutine ha_prepare_state

  function ha_sym_curve(trans_in, temp_in, eta_in, wi, wf, voltages) result(currents)
    implicit none
    real, intent(in) :: trans_in, temp_in, eta_in, wi, wf
    real, intent(in) :: voltages(:)
    real :: currents(size(voltages))

    complex :: Atol, Rtol
    complex, external :: zint
    real :: temp_eval, thop_eval, eta_eval
    integer :: ierr, k
    integer :: nan_eval

    Atol = (1.e-8, 1.0)
    Rtol = (1.e-6, 1.0)
    ierr = 0

    do k = 1, size(voltages)
       call ha_prepare_state(trans_in, temp_in, eta_in, voltages(k), &
            temp_eval, thop_eval, eta_eval, nan_eval)
       currents(k) = real(zint(wi, wf, Atol, Rtol, ierr, voltages(k), &
            temp_eval, thop_eval, eta_eval, nan_eval))
    end do
  end function ha_sym_curve

end module ha_sym_api
