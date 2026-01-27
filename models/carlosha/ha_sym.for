c                      PROGRAM IV


c   ***************************************************************
c
c   This program computes the dc current in a single-channel
c   superconducting point contact. 
c    
c   [see Cuevas et al., PRB (1996).]
c   ***************************************************************


c
c VARIABLE DEFINITION
c
c  NOTE (2025-12): This file has been adapted so that the former
c  standalone program can be called from Python via f2py.
c  The new entry point is `ha_sym_curve`, which takes inputs
c  analogous to the parameters in iv.in and returns the I(V)
c  curve without any file I/O.

cc Convenience multi-voltage wrapper (subroutine; f2py returns array)
c
c Fixed-form F77 array-valued functions with runtime length are not
c reliably wrapped by f2py. This subroutine version is intended to
c provide the clean Python call style:
c
c   currents = ha_sym_curve(trans, temp, eta, wi, wf, voltages)

      subroutine ha_sym_curve(trans_in,temp_in,eta_in,wi,wf,
     &                        voltages,currents,npts)

!f2py intent(in) trans_in,temp_in,eta_in,wi,wf,voltages
!f2py intent(out) currents
!f2py depend(voltages) npts = len(voltages)

       implicit real (a-h,o-z)
       parameter (ns=520)

       integer npts, k, ierr
       integer nan
       real trans_in,temp_in,eta_in,wi,wf
       real voltages(npts), currents(npts)
       real eta
       complex ui,zint,Atol,Rtol

       common/argum/v,temp,thop
       common/dynes/eta
       common/dim/nan

       external inv

       pi=4.0*atan(1.0)
       ui=(0.0,1.0)
       Atol=(1.e-8,1.0)
       Rtol=(1.e-6,1.0)
       ierr=0

       temp = temp_in
       if (temp.eq.0.0) temp = 1.e-7

       eta = eta_in
       if (eta.le.0.0) eta = 1.e-6

       trans = trans_in
       if (trans.le.0.0) then
          thop = 0.0
       else
          thop = sqrt((2.0-trans-2.0*sqrt(1.0-trans))/trans)
       end if

       DO k=1,npts
          v = voltages(k)

          if (abs(v).lt.1.e-12) then
              nan = ns
          else
              nan = int(2.0/abs(v))
              if (mod(nan,2).eq.0) then
                  nan = nan + 7
              else
                  nan = nan + 6
              end if
          end if

          if (nan.gt.ns) nan = ns

          currents(k) = real(zint(wi,wf,Atol,Rtol,ierr))
       END DO

       return
       end

c Convenience single-voltage wrapper (function)
      real function ha_sym_single(trans_in,temp_in,eta_in,wi,wf,v_in)

       implicit real (a-h,o-z)
       parameter (ns=520)

       integer nan,ierr,nmax
       real wi,wf,trans_in,temp_in,eta_in,v_in
       real eta
       complex ui,zint,Atol,Rtol

       common/argum/v,temp,thop
       common/dynes/eta
       common/dim/nan

       external inv

       pi=4.0*atan(1.0)
       ui=(0.0,1.0)
       Atol=(1.e-8,1.0)
       Rtol=(1.e-6,1.0)
       ierr=0

       temp = temp_in
       if (temp.eq.0.0) temp = 1.e-7

       eta = eta_in
       if (eta.le.0.0) eta = 1.e-6

       trans = trans_in
       if (trans.le.0.0) then
          thop = 0.0
       else
          thop = sqrt((2.0-trans-2.0*sqrt(1.0-trans))/trans)
       end if

       v = v_in

       if (abs(v).lt.1.e-12) then
          nan = ns
       else
          nan = int(2.0/abs(v))
          if (mod(nan,2).eq.0) then
              nan = nan + 7
          else
              nan = nan + 6
          end if
       end if
       if (nan.gt.ns) nan = ns

       ha_sym_single = real(zint(wi,wf,Atol,Rtol,ierr))

       return
       end

*************************************************************************
c
c FUNCTION zintegrand (calculation of the current density)
c
******************
        complex function zintegrand(w)
        implicit real (a-h,o-z)
        parameter (ns=520)
        integer nan
        real w,curr,tau3(2,2)
        real eta
        complex ui,wwj,omega,g0lr(2,2,-ns:ns),g0rr(2,2,-ns:ns),
     &          g0la(2,2,-ns:ns),g0ra(2,2,-ns:ns),
     &          g0kl(2,2,-ns:ns),g0kr(2,2,-ns:ns),
     &          er(2,2,-ns:ns),vpr(2,2,-ns:ns),vmr(2,2,-ns:ns),
     &          ea(2,2,-ns:ns),vpa(2,2,-ns:ns),vma(2,2,-ns:ns),
     &          adr(2,2,1:ns),air(2,2,-ns:-1),
     &          ada(2,2,1:ns),aia(2,2,-ns:-1),
     &          cpr(2,2),cmr(2,2),cpa(2,2),cma(2,2),
     &          aux1r(2,2),aux2r(2,2),aux3r(2,2),aux4r(2,2),
     &          aux1a(2,2),aux2a(2,2),aux3a(2,2),aux4a(2,2),
     &          tr(2,2,-ns:ns),ta(2,2,-ns:ns),tx(2,2),ty(2,2)

        common/argum/v,temp,thop
        common/dim/nan
        common/dynes/eta

        pi=4.0*atan(1.0)
        ui=(0.0,1.0)
        t2 = thop**2.0
        tau3(1,1) = 1.0
        tau3(2,2) = -1.0
        delta = 1.0
c
c Now the program computes the surface Green functions:
c Note: introduce the GF without the pi factor.
c
        do j=-nan-2,nan+2
           wj = w + float(j)*v
           wwj = wj + ui*eta
           omega = csqrt(delta**2 - wwj**2)

           g0lr(1,1,j) = -wwj/omega
           g0lr(1,2,j) = delta/omega
           g0lr(2,1,j) = -delta/omega
           g0lr(2,2,j) = wwj/omega

           g0la(1,1,j) = conjg(g0lr(1,1,j))
           g0la(1,2,j) = -conjg(g0lr(2,1,j))
           g0la(2,1,j) = -conjg(g0lr(1,2,j))
           g0la(2,2,j) = conjg(g0lr(2,2,j))

           g0rr(1,1,j) = g0lr(1,1,j)
           g0rr(1,2,j) = g0lr(1,2,j)
           g0rr(2,1,j) = g0lr(2,1,j)
           g0rr(2,2,j) = g0lr(2,2,j)

           g0ra(1,1,j) = conjg(g0rr(1,1,j))
           g0ra(1,2,j) = -conjg(g0rr(2,1,j))
           g0ra(2,1,j) = -conjg(g0rr(1,2,j))
           g0ra(2,2,j) = conjg(g0rr(2,2,j))

           fact = tanh(0.5*wj/temp)
           do k=1,2
              do l=1,2
                 g0kl(k,l,j) = (g0lr(k,l,j)-g0la(k,l,j))*fact
                 g0kr(k,l,j) = (g0rr(k,l,j)-g0ra(k,l,j))*fact
              end do
           end do
        end do
c
c Parameters of the algebraic system of the T-matrix components:
c
        DO j=-nan,nan,2

           er(1,1,j) = 1.0 - t2*g0rr(1,1,j+1)*g0lr(1,1,j)
           er(1,2,j) = -t2*g0rr(1,1,j+1)*g0lr(1,2,j)
           er(2,1,j) = -t2*g0rr(2,2,j-1)*g0lr(2,1,j)
           er(2,2,j) = 1.0 - t2*g0rr(2,2,j-1)*g0lr(2,2,j)

           ea(1,1,j) = 1.0 - t2*g0ra(1,1,j+1)*g0la(1,1,j)
           ea(1,2,j) = -t2*g0ra(1,1,j+1)*g0la(1,2,j)
           ea(2,1,j) = -t2*g0ra(2,2,j-1)*g0la(2,1,j)
           ea(2,2,j) = 1.0 - t2*g0ra(2,2,j-1)*g0la(2,2,j)

           vpr(1,1,j) = t2*g0rr(1,2,j+1)*g0lr(2,1,j+2)
           vpr(1,2,j) = t2*g0rr(1,2,j+1)*g0lr(2,2,j+2)
           vpr(2,1,j) = (0.0,0.0)
           vpr(2,2,j) = (0.0,0.0)

           vpa(1,1,j) = t2*g0ra(1,2,j+1)*g0la(2,1,j+2)
           vpa(1,2,j) = t2*g0ra(1,2,j+1)*g0la(2,2,j+2)
           vpa(2,1,j) = (0.0,0.0)
           vpa(2,2,j) = (0.0,0.0)

           vmr(1,1,j) = (0.0,0.0)
           vmr(1,2,j) = (0.0,0.0)
           vmr(2,1,j) = t2*g0rr(2,1,j-1)*g0lr(1,1,j-2)
           vmr(2,2,j) = t2*g0rr(2,1,j-1)*g0lr(1,2,j-2)

           vma(1,1,j) = (0.0,0.0)
           vma(1,2,j) = (0.0,0.0)
           vma(2,1,j) = t2*g0ra(2,1,j-1)*g0la(1,1,j-2)
           vma(2,2,j) = t2*g0ra(2,1,j-1)*g0la(1,2,j-2)

        END DO
c
c SELF-ENERGIES:
c
        do i=1,2
           do j=1,2
              aux1r(i,j) = er(i,j,nan)
              aux3r(i,j) = er(i,j,-nan)

              aux1a(i,j) = ea(i,j,nan)
              aux3a(i,j) = ea(i,j,-nan)
           end do
        end do
        call inv(aux1r,aux2r,2,2)
        call inv(aux3r,aux4r,2,2)
        call inv(aux1a,aux2a,2,2)
        call inv(aux3a,aux4a,2,2)
        do i=1,2
           do j=1,2
              adr(i,j,nan) = aux2r(i,j)
              air(i,j,-nan) = aux4r(i,j)

              ada(i,j,nan) = aux2a(i,j)
              aia(i,j,-nan) = aux4a(i,j)
           end do
        end do
c
        DO i=nan-2,1,-2

           do k=1,2
              do l=1,2
                 aux1r(k,l) = er(k,l,i)
                 aux3r(k,l) = er(k,l,-i)
                 aux1a(k,l) = ea(k,l,i)
                 aux3a(k,l) = ea(k,l,-i)
                 do i1=1,2
                    do i2=1,2
                       aux1r(k,l) = aux1r(k,l) - 
     &                          vpr(k,i1,i)*adr(i1,i2,i+2)*vmr(i2,l,i+2)
                       aux3r(k,l) = aux3r(k,l) -      
     &                       vmr(k,i1,-i)*air(i1,i2,-i-2)*vpr(i2,l,-i-2)

                       aux1a(k,l) = aux1a(k,l) - 
     &                          vpa(k,i1,i)*ada(i1,i2,i+2)*vma(i2,l,i+2)
                       aux3a(k,l) = aux3a(k,l) -      
     &                       vma(k,i1,-i)*aia(i1,i2,-i-2)*vpa(i2,l,-i-2)
                    end do
                 end do
              end do
           end do
           call inv(aux1r,aux2r,2,2)
           call inv(aux3r,aux4r,2,2)
           call inv(aux1a,aux2a,2,2)
           call inv(aux3a,aux4a,2,2)
           do ia=1,2
              do ib=1,2
                 adr(ia,ib,i) = aux2r(ia,ib)
                 air(ia,ib,-i) = aux4r(ia,ib)
                 ada(ia,ib,i) = aux2a(ia,ib)
                 aia(ia,ib,-i) = aux4a(ia,ib)
              end do
           end do

        END DO
c
c Closed system for T_{1,0} and T_{-1,0}:
c

c tx --> v_{1,0}  and ty --> v_{-1,0}

        do k=1,2
           do l=1,2
              tx(k,l) = (0.0,0.0)
              ty(k,l) = (0.0,0.0)
           end do
        end do
        tx(2,2) = thop
        ty(1,1) = thop

        do k=1,2
           do l=1,2
              aux1r(k,l) = er(k,l,1) 
              aux1a(k,l) = ea(k,l,1)
              cpr(k,l) = tx(k,l)
              cpa(k,l) = tx(k,l)
              do i1=1,2
                 do i2=1,2
                    aux1r(k,l) = aux1r(k,l) 
     &              - vpr(k,i1,1)*adr(i1,i2,3)*vmr(i2,l,3) 
     &              - vmr(k,i1,1)*air(i1,i2,-1)*vpr(i2,l,-1)

                    aux1a(k,l) = aux1a(k,l) 
     &              - vpa(k,i1,1)*ada(i1,i2,3)*vma(i2,l,3) 
     &              - vma(k,i1,1)*aia(i1,i2,-1)*vpa(i2,l,-1)

                    cpr(k,l) = cpr(k,l) +
     &                vmr(k,i1,1)*air(i1,i2,-1)*ty(i2,l)

                    cpa(k,l) = cpa(k,l) +
     &                vma(k,i1,1)*aia(i1,i2,-1)*ty(i2,l)
                 end do
              end do
           end do
        end do
        call inv(aux1r,aux2r,2,2)
        call inv(aux1a,aux2a,2,2)
        do i=1,2
           do j=1,2
              tr(i,j,1) = (0.0,0.0)
              ta(i,j,1) = (0.0,0.0)
              do i1=1,2
                 tr(i,j,1) = tr(i,j,1) + aux2r(i,i1)*cpr(i1,j)
                 ta(i,j,1) = ta(i,j,1) + aux2a(i,i1)*cpa(i1,j)
              end do
           end do
        end do

        do k=1,2
           do l=1,2
              cmr(k,l) = ty(k,l)
              cma(k,l) = ty(k,l)
              do i1=1,2
                 cmr(k,l) = cmr(k,l) + vpr(k,i1,-1)*tr(i1,l,1)
                 cma(k,l) = cma(k,l) + vpa(k,i1,-1)*ta(i1,l,1)
              end do
           end do
        end do

        do i=1,2
           do j=1,2
              tr(i,j,-1) = (0.0,0.0)
              ta(i,j,-1) = (0.0,0.0)
              do i1=1,2
                 tr(i,j,-1) = tr(i,j,-1) + air(i,i1,-1)*cmr(i1,j)
                 ta(i,j,-1) = ta(i,j,-1) + aia(i,i1,-1)*cma(i1,j)
              end do
           end do
        end do
c
c Calculation of the rest of the T's using the recursive relations:
c
        DO j=3,nan,2
           do k=1,2
              do l=1,2
                 tr(k,l,j) = (0.0,0.0)
                 tr(k,l,-j) = (0.0,0.0)
                 ta(k,l,j) = (0.0,0.0)
                 ta(k,l,-j) = (0.0,0.0)
                 do i1=1,2
                    do i2=1,2
                       tr(k,l,j) = tr(k,l,j) + 
     &                 adr(k,i1,j)*vmr(i1,i2,j)*tr(i2,l,j-2)

                       ta(k,l,j) = ta(k,l,j) + 
     &                 ada(k,i1,j)*vma(i1,i2,j)*ta(i2,l,j-2)

                       tr(k,l,-j) = tr(k,l,-j) + 
     &                 air(k,i1,-j)*vpr(i1,i2,-j)*tr(i2,l,-j+2)

                       ta(k,l,-j) = ta(k,l,-j) + 
     &                 aia(k,i1,-j)*vpa(i1,i2,-j)*ta(i2,l,-j+2)
                    end do
                 end do
              end do
           end do
        END DO

c Current density:

        curr = 0.0
         do j=-nan,nan,2
           do i1=1,2
               do i2=1,2
                  do i3=1,2
                     do i4=1,2
                       curr = curr + 
     &   real(tau3(i1,i1)*tr(i1,i2,j)*g0kr(i2,i3,0)*tau3(i3,i3)*
     &   conjg(tr(i4,i3,j))*tau3(i4,i4)*g0la(i4,i1,j) +
     &   tau3(i1,i1)*g0rr(i1,i2,0)*tau3(i2,i2)*
     &   conjg(ta(i3,i2,j))*tau3(i3,i3)*g0kl(i3,i4,j)*
     &   ta(i4,i1,j))

                     end do
                  end do
              end do
           end do
        end do

        curr = curr/2.d0 
        zintegrand = curr

c        zintegrand=tr(1,1,1)

        return
        end
C ***************************************************************
      subroutine inv(a,ainv,n,ndim)
      complex a(ndim,ndim),ainv(ndim,ndim),det
      integer n,ndim

      det = a(1,1)*a(2,2) - a(1,2)*a(2,1)
      ainv(1,1) = a(2,2)/det
      ainv(1,2) = -a(1,2)/det
      ainv(2,1) = -a(2,1)/det
      ainv(2,2) = a(1,1)/det

      return
      end
C ***************************************************************
c------------------------------------------------------------------------
c
      complex function zint(llim,ulim,Atol,Rtol,ierr)
c
c------------------------------------------------------------------------
c
      implicit none
      integer ierr
      complex Atol, Rtol
      real llim, ulim
c
      integer stksz
c      parameter (stksz = 1024)
      parameter (stksz = 4096)
      dimension xstk(stksz), zstk(stksz)
      real xstk
      complex zstk
c
      integer itask, istk
      complex lz, mz, uz, zres, zintegrand
      real lx, mx, ux
      real  zero, two
      parameter (zero=0.0, two=2.0)
c
c------------------------------------------------------------------------
c
c   **Set some things to zero.**
      istk = 0
      ierr = 0
      zint = cmplx(zero,zero)
c
c   **Get the three points for the first Simpson integral.**

      lx = llim
      mx = (llim + ulim) / two
      ux = ulim
      lz = zintegrand(lx)
      mz = zintegrand(mx)
      uz = zintegrand(ux)
c
c   **Initialize the stacks.**
      xstk(1) = lx
      xstk(2) = mx
      xstk(3) = ux
      zstk(1) = lz
      zstk(2) = mz
      zstk(3) = uz
c
 10   continue
c
c   **Call the recursive part.**
      call zintrp(zres,lx,lz,mx,mz,ux,uz,istk,xstk,zstk,Atol,Rtol,itask)
c
c   **Check if we've exceeded the stack size.**
      if (3*(istk+1) .ge. stksz) then
        write (*,*) 'zint_: Stack size exceeded!'
        close(80)
        close(81)
        stop
      end if
c
c   **ierr keeps track of how big the stack gets.**
      if (istk .gt. ierr)  ierr=istk
c
c   **See if the current branch has converged.**
      if (itask .eq. 0) then
        zint = zint + zres
      else
        goto 10
      end if
c
c   **If the stack size is not zero, we have more integrating to do.**
c   **Pop the top three guys off the stack and integrate over them. **
      if (istk .ne. 0) then
        istk = istk - 1
        lx = xstk(3*istk+1)
        mx = xstk(3*istk+2)
        ux = xstk(3*istk+3)
        lz = zstk(3*istk+1)
        mz = zstk(3*istk+2)
        uz = zstk(3*istk+3)
        goto 10
      end if
c
c666  write(*,1000) ' Zint=',zint
c1000 format(a,2(x,e12.6))
 666  return
      end
c
c**********************************************************************
c
      subroutine zintrp(zres,lx,lz,mx,mz,ux,uz,istk,xstk,zstk,
     &                  Atol,Rtol,itask)
c
      implicit none
      integer stksz
      parameter (stksz = 1024)

      integer istk, itask
      complex zres, lz, mz, uz, Atol, Rtol, zstk(stksz)
      real lx, mx, ux, xstk(stksz)
c
      complex ctest, lmz, umz, zintegrand
      real dx, lmx, umx, tmpr, tmpi
      real two, three, four      
      parameter (two=2.0)
      parameter (three=3.0)
      parameter (four=4.0)
c
c
c   **Do the initial three-point Simpson's rule for reference.
      dx = (ux-lx) / two
      ctest = dx*(lz + four*mz + uz) / three
c
c   **Do the five-point Simpson's rule as a check.
      dx = dx / two
      lmx = (lx + mx) / two
      lmz = zintegrand(lmx)
      umx = (ux + mx) / two
      umz = zintegrand(umx)
      zres = dx*(lz + two*mz + four*(lmz+umz) + uz) / three
c
c   **Check the absolute tolerance and then the relative tolerance.**
      if ( abs(real(zres-ctest)) .lt. real(Atol) ) then
        if ( abs(aimag(zres-ctest)) .lt. aimag(Atol) ) then
          itask = 0
        else
          tmpi = two*abs(aimag(zres-ctest)) / abs(aimag(zres+ctest))
          if ( tmpi .lt. aimag(Rtol) ) then
            itask = 0
          else
            itask = 1
          end if
        end if
      else
        tmpr = two*abs(real(zres-ctest))  / abs(real(zres+ctest))
        if ( tmpr .lt. real(Rtol) ) then
          if ( abs(aimag(zres-ctest)) .lt. aimag(Atol) ) then
            itask = 0
          else
            tmpi = two*abs(aimag(zres-ctest)) / abs(aimag(zres+ctest))
            if ( tmpi .lt. aimag(Rtol) ) then
              itask = 0
            else
              itask = 1
            end if
          end if
        else
          itask = 1
        end if
      end if
c
c   **If we must go on, push the upper three guys onto the stack and **
c   **integrate over the lower three.                                **
      if (itask .eq. 1) then
        xstk(3*istk+1) = mx
        xstk(3*istk+2) = umx
        xstk(3*istk+3) = ux
        zstk(3*istk+1) = mz
        zstk(3*istk+2) = umz
        zstk(3*istk+3) = uz
        istk = istk + 1
        ux = mx
        mx = lmx
        uz = mz
        mz = lmz
      end if
c
      return
      end
c
c------------------------------------------------------------------------
