c                      PROGRAM HARMONICS (asymmetric)


c   ***************************************************************
c           THIS PROGRAM CALCULATES THE I-V CURVES IN A
c     SUPERCONDUCTING QUANTUM POINT CONTACT WITH ONLY ONE CHANNEL.
c
c       (FOR EVERY VOLTAGE,TEMPERATURE,TRANSMISSION AND HARMONIC)
c
c              Cuevas et al. PRB 54, 7366 (1996).

c            ASYMMETRIC CASE: DELTAL  DELTAR
c   ***************************************************************


c
c VARIABLE DEFINITION
c
       implicit real*8 (a-h,o-z)
       integer n,m,iv,ierr
       complex*16 ui,curri,curr,zint,Atol,Rtol
      
       common/param/dmvt1,dmvt2,eta1,eta2
       common/dim/n,m
       common/hop/t,t2,temp,v
c
c READING THE DATA SET
c ********************
c
c The program reads the input parameters from a external file
c
c List of parameter:
c
c  trans= normal transmission coefficient (can be varied from 0 to 1).
c         The normal linear conductance of this single-mode contact
c         at zero temperature is given by:   G=(2e**2/h)*trans

c  temp= temperature/Delta1(zero temperature)

c  eta = Dynes parameter 

c  vi= Initial value of the voltage.

c  vf= Final value of the voltage.

c  vstep= Step in the voltage values.
c
c  ratio = Delta2/Delta1   (Delta2 < Delta1)

       m = 0
       read(5,*)temp
       read(5,*)d1,d2
       read(5,*)trans
       read(5,*)eta1,eta2
       read(5,*)vi,vf,vstep

c       vi=-vf

       eta1 = eta1/d1
       eta2 = eta2/d1
c temp in units of d1
       temp = 0.08617d0*temp/d1
c ratio
       ratio = d2/d1
c voltage in reduced units
       vi = vi/d1
       vf = vf/d1
       vstep = vstep/d1
       iv = int((vf-vi)/vstep)
c hopping element
       t = dsqrt((2.d0-trans-2.d0*dsqrt(1.d0-trans))/trans)
       t2=t*t
c some constants
       pi=4.d0*datan(1.d0)
       ui=(0.d0,1.d0)
c precision of the energy integration
       Atol=(1.d-12,1.d-12)
       Rtol=(1.d-9,1.d-9)
       ierr=0

       if(temp.eq.0.d0) temp = 1.d-7
       if(eta.eq.0.d0) eta = 1.d-6
c limits of the energy integration
       wf = dabs(vf)*5.d0
       wi = -wf

c Subroutine gap calculates the gap at finite temperature.
       call gap(temp,dmvt1)
       call gap(temp/ratio,dmvt2)
       write(*,*)'#','gap1 = ',dmvt1,' gap2 = ',dmvt2
       dmvt2 = dmvt2*ratio
c       stop

c Voltage loop

       DO k=1,iv+1
          v = vi + dfloat(k)*vstep
          n = int(2.0/dabs(v))
          if (mod(n,2).eq.0) then
              n = n + 6
          else
              n = n + 7
          end if
          if (dabs(v).lt.0.002d0) then
              curri = 0.d0
          else
              curri = zint(wi,wf,Atol,Rtol,ierr)
          end if
c Write voltage in mV and current in nA
          write(*,*)v*d1,dreal(curri)*d1*77.48d0
c          write(*,*)v*d1,dreal(curri)*d1/trans
       END DO

       END

*************************************************************************
*************************************************************************
c
c SUBROUTINE GREEN0 
c
********************
c
c This sobroutine computes the uncoupled BCS Green Functions of the leads
c
        subroutine green0(z,gsl,gsr)
        implicit real*8 (a-h,o-z)
        common/param/dmvl,dmvr,etal,etar
        complex*16 u,gsl(2,2),gsr(2,2)

        if (dmvl.eq.0) then
	    gsl(1,1) = (0.d0,1.d0)
	    gsl(1,2) = (0.d0,0.d0)
	    gsl(2,1) = (0.d0,0.d0)
	    gsl(2,2) = (0.d0,1.d0)
        else
            u = dcmplx(z,-etal)/dmvl
	    gsl(1,1) = -u/cdsqrt(1.d0-u**2)
            gsl(1,2) = 1.d0/cdsqrt(1.d0-u**2)
	    gsl(2,1) = gsl(1,2)
	    gsl(2,2) = gsl(1,1)
        end if
        if (dmvr.eq.0) then
	    gsl(1,1) = (0.d0,1.d0)
	    gsl(1,2) = (0.d0,0.d0)
	    gsl(2,1) = (0.d0,0.d0)
	    gsl(2,2) = (0.d0,1.d0)
        else
            u = dcmplx(z,-etar)/dmvr
	    gsr(1,1) = -u/cdsqrt(1.d0-u**2)
            gsr(1,2) = 1.d0/cdsqrt(1.d0-u**2)
	    gsr(2,1) = gsr(1,2)
	    gsr(2,2) = gsr(1,1)
        end if
        return
        end
************************************************************************
************************************************************************
c
c FUNCTION zintegrand (calculation of the current density)
c
******************
        complex*16 function zintegrand(w)
        implicit real*8 (a-h,o-z)
        parameter (ns=5000,np=10)
        integer n,m
        complex*16 e(2,2,-ns:ns),vr(2,2,-ns:ns,-1:1),
     +          s(2,2),det,ad(2,2,0:ns),ai(2,2,-ns:np),b(2,2)
       complex*16 ui,curr,gsl(2,2),gsr(2,2),g1,g3,g4,
     +            g0l(2,2,-ns:ns),gpml(2,2,-ns:ns),gmpl(2,2,-ns:ns),
     +            g0r(2,2,-ns:ns),gpmr(2,2,-ns:ns),gmpr(2,2,-ns:ns),
     +            gr(2,2,-ns:ns,-2:np),glr(2,2,-ns:ns,-2:np)
        common/dim/n,mmax
        common/hop/t,t2,temp,v

        ui=(0.d0,1.d0)
c
c Now the program calculates the uncoupled Green Functions
c
          do j=-n-1,n+1
             wj=w+dfloat(j)*v
             call green0(wj,gsl,gsr)
             g0l(1,1,j)=gsl(1,1)
             g0l(1,2,j)=gsl(1,2)
             g0l(2,1,j)=gsl(2,1)
             g0l(2,2,j)=gsl(2,2)
	     g0r(1,1,j)=gsr(1,1)
	     g0r(1,2,j)=gsr(1,2)
	     g0r(2,1,j)=gsr(2,1)
	     g0r(2,2,j)=gsr(2,2)

             fermi = 0.5d0*(1.d0-tanh(0.5d0*wj/temp)) 

             gpml(1,1,j)=2.d0*ui*dimag(gsl(1,1))*fermi
             gpml(1,2,j)=2.d0*ui*dimag(gsl(1,2))*fermi
             gpml(2,1,j)=gpml(1,2,j)
             gpml(2,2,j)=gpml(1,1,j)
             gmpl(1,1,j)=2.d0*ui*dimag(gsl(1,1))*(fermi-1.d0)
             gmpl(1,2,j)=2.d0*ui*dimag(gsl(1,2))*(fermi-1.d0)
             gmpl(2,1,j)=gmpl(1,2,j)
             gmpl(2,2,j)=gmpl(1,1,j)
             gpmr(1,1,j)=2.d0*ui*dimag(gsr(1,1))*fermi
             gpmr(1,2,j)=2.d0*ui*dimag(gsr(1,2))*fermi
             gpmr(2,1,j)=gpmr(1,2,j)
             gpmr(2,2,j)=gpmr(1,1,j)
             gmpr(1,1,j)=2.d0*ui*dimag(gsr(1,1))*(fermi-1.d0)
             gmpr(1,2,j)=2.d0*ui*dimag(gsr(1,2))*(fermi-1.d0)
             gmpr(2,1,j)=gmpr(1,2,j)
             gmpr(2,2,j)=gmpr(1,1,j)
          end do

c
c SYSTEM PARAMETERS
c
        do 10 j=-n,n,2

c E(N)--> 1-E(N)

           e(1,1,j)=1.d0-t2*g0r(1,1,j)*g0l(1,1,j-1)
           e(1,2,j)=-t2*g0r(1,2,j)*g0l(2,2,j+1)
           e(2,1,j)=-t2*g0r(2,1,j)*g0l(1,1,j-1)
           e(2,2,j)=1.d0-t2*g0r(2,2,j)*g0l(2,2,j+1)

c V(N,N-2)--> V(N,-1)

           vr(1,2,j,-1)=-t2*g0r(1,1,j)*g0l(1,2,j-1)
           vr(2,2,j,-1)=-t2*g0r(2,1,j)*g0l(1,2,j-1)

c V(N,N+2)--> V(N,1)

           vr(1,1,j,1)=-t2*g0r(1,2,j)*g0l(2,1,j+1)
           vr(2,1,j,1)=-t2*g0r(2,2,j)*g0l(2,1,j+1)

 10     continue
c
c SELF-ENERGIES
c
c RIGHT and LEFT

        det=e(1,1,n)*e(2,2,n) - e(1,2,n)*e(2,1,n)
        ad(1,1,n)=e(2,2,n)/det
        ad(1,2,n)=-e(1,2,n)/det
        ad(2,1,n)=-e(2,1,n)/det
        ad(2,2,n)=e(1,1,n)/det
c
        det=e(1,1,-n)*e(2,2,-n) - e(1,2,-n)*e(2,1,-n)
        ai(1,1,-n)=e(2,2,-n)/det
        ai(1,2,-n)=-e(1,2,-n)/det
        ai(2,1,-n)=-e(2,1,-n)/det
        ai(2,2,-n)=e(1,1,-n)/det
c
        do 20 i=n-2,0,-2

           s(1,1)=e(1,1,i)
           s(1,2)=e(1,2,i)-vr(1,1,i,1)*ad(1,1,i+2)*vr(1,2,i+2,-1)
     +                    -vr(1,1,i,1)*ad(1,2,i+2)*vr(2,2,i+2,-1)           
           s(2,1)=e(2,1,i)
           s(2,2)=e(2,2,i)-vr(2,1,i,1)*ad(1,1,i+2)*vr(1,2,i+2,-1)
     +                    -vr(2,1,i,1)*ad(1,2,i+2)*vr(2,2,i+2,-1)
           det=s(1,1)*s(2,2) - s(1,2)*s(2,1)
           ad(1,1,i)=s(2,2)/det
           ad(1,2,i)=-s(1,2)/det
           ad(2,1,i)=-s(2,1)/det
           ad(2,2,i)=s(1,1)/det

 20     continue
c
        do 30 i=n-2,-mmax,-2

           s(1,1)=e(1,1,-i)-vr(1,2,-i,-1)*ai(2,1,-i-2)*vr(1,1,-i-2,1)
     +                     -vr(1,2,-i,-1)*ai(2,2,-i-2)*vr(2,1,-i-2,1)
           s(1,2)=e(1,2,-i)
           s(2,1)=e(2,1,-i)-vr(2,2,-i,-1)*ai(2,1,-i-2)*vr(1,1,-i-2,1)
     +                     -vr(2,2,-i,-1)*ai(2,2,-i-2)*vr(2,1,-i-2,1)
           s(2,2)=e(2,2,-i)
           det=s(1,1)*s(2,2) - s(1,2)*s(2,1)
           ai(1,1,-i)=s(2,2)/det
           ai(1,2,-i)=-s(1,2)/det
           ai(2,1,-i)=-s(2,1)/det
           ai(2,2,-i)=s(1,1)/det

 30     continue
c
c SOLUTION OF THE 2x2 SYSTEMS OBTAINED PROJECTING
c
        do m=-2,mmax,2

           b(1,1)=e(1,1,m)-vr(1,2,m,-1)*ai(2,1,m-2)*vr(1,1,m-2,1)
     +                    -vr(1,2,m,-1)*ai(2,2,m-2)*vr(2,1,m-2,1)
           b(1,2)=e(1,2,m)-vr(1,1,m,1)*ad(1,1,m+2)*vr(1,2,m+2,-1)
     +                    -vr(1,1,m,1)*ad(1,2,m+2)*vr(2,2,m+2,-1)
           b(2,1)=e(2,1,m)-vr(2,2,m,-1)*ai(2,1,m-2)*vr(1,1,m-2,1)
     +                    -vr(2,2,m,-1)*ai(2,2,m-2)*vr(2,1,m-2,1)
           b(2,2)=e(2,2,m)-vr(2,1,m,1)*ad(1,1,m+2)*vr(1,2,m+2,-1)
     +                    -vr(2,1,m,1)*ad(1,2,m+2)*vr(2,2,m+2,-1)
           det=b(1,1)*b(2,2) - b(1,2)*b(2,1)

           gr(1,1,m,m)=(b(2,2)*g0r(1,1,m)-b(1,2)*g0r(2,1,m))/det
           gr(1,2,m,m)=(b(2,2)*g0r(1,2,m)-b(1,2)*g0r(2,2,m))/det
           gr(2,1,m,m)=(b(1,1)*g0r(2,1,m)-b(2,1)*g0r(1,1,m))/det
           gr(2,2,m,m)=(b(1,1)*g0r(2,2,m)-b(2,1)*g0r(1,2,m))/det

c
c CALCULATION OF THE REST OF THE GR GREEN FUNCTIONS USING THE RECURSIVE
c                         RELATIONS
c

           do k=m+2,n,2
              gr(1,1,k,m)=ad(1,1,k)*vr(1,2,k,-1)*gr(2,1,k-2,m)+
     +                    ad(1,2,k)*vr(2,2,k,-1)*gr(2,1,k-2,m)
              gr(1,2,k,m)=ad(1,1,k)*vr(1,2,k,-1)*gr(2,2,k-2,m)+
     +                    ad(1,2,k)*vr(2,2,k,-1)*gr(2,2,k-2,m)
              gr(2,1,k,m)=ad(2,1,k)*vr(1,2,k,-1)*gr(2,1,k-2,m)+
     +                    ad(2,2,k)*vr(2,2,k,-1)*gr(2,1,k-2,m)
              gr(2,2,k,m)=ad(2,1,k)*vr(1,2,k,-1)*gr(2,2,k-2,m)+
     +                    ad(2,2,k)*vr(2,2,k,-1)*gr(2,2,k-2,m)
           end do

           do k=-m+2,n,2
              gr(1,1,-k,m)=ai(1,1,-k)*vr(1,1,-k,1)*gr(1,1,-k+2,m)+
     +                     ai(1,2,-k)*vr(2,1,-k,1)*gr(1,1,-k+2,m)
              gr(1,2,-k,m)=ai(1,1,-k)*vr(1,1,-k,1)*gr(1,2,-k+2,m)+
     +                     ai(1,2,-k)*vr(2,1,-k,1)*gr(1,2,-k+2,m)
              gr(2,1,-k,m)=ai(2,1,-k)*vr(1,1,-k,1)*gr(1,1,-k+2,m)+
     +                     ai(2,2,-k)*vr(2,1,-k,1)*gr(1,1,-k+2,m)
              gr(2,2,-k,m)=ai(2,1,-k)*vr(1,1,-k,1)*gr(1,2,-k+2,m)+
     +                     ai(2,2,-k)*vr(2,1,-k,1)*gr(1,2,-k+2,m)
           end do
        
        end do

c
c CALCULATION OF THE GLR GREEN FUNCTIONS 
c

        do m=-2,mmax,2

           do j=-n+1,n-1,2
              glr(1,1,j,m)=t*(g0l(1,1,j)*gr(1,1,j+1,m)-
     +                        g0l(1,2,j)*gr(2,1,j-1,m))
              glr(1,2,j,m)=t*(g0l(1,1,j)*gr(1,2,j+1,m)-
     +                        g0l(1,2,j)*gr(2,2,j-1,m))
              glr(2,1,j,m)=t*(g0l(2,1,j)*gr(1,1,j+1,m)-
     +                        g0l(2,2,j)*gr(2,1,j-1,m))
              glr(2,2,j,m)=t*(g0l(2,1,j)*gr(1,2,j+1,m)-
     +                        g0l(2,2,j)*gr(2,2,j-1,m))
           end do

        end do

c
c Calculation of the G(+,-) y G(-,+)
c
c g1--> G(+,-)(RR)(1,1)(0,m)
c

          g1=(0.d0,0.d0)
          if (mmax.eq.0) then
              g1= gpmr(1,1,0)
          end if 

          g1= g1 + t*(gpmr(1,1,0)*glr(1,1,-1,mmax) - 
     &                gpmr(1,2,0)*glr(2,1,1,mmax)  +
     &                dconjg(glr(1,1,mmax-1,0))*gpmr(1,1,mmax) -
     &                dconjg(glr(2,1,mmax+1,0))*gpmr(2,1,mmax))            
c
c g3--> G(+,-)(RR)(2,1)(-2,m)
c

          g3= t*(gpmr(2,1,-2)*glr(1,1,-3,mmax) -
     &           gpmr(2,2,-2)*glr(2,1,-1,mmax)  +
     &           dconjg(glr(1,2,mmax-1,-2))*gpmr(1,1,mmax) -
     &           dconjg(glr(2,2,mmax+1,-2))*gpmr(2,1,mmax))
c
c g4--> G(+,-)(RR)(1,2)(0,m-2)
c

          g4=(0.d0,0.d0)
          if (mmax.eq.2) then
              g4= gpmr(1,2,0)
          end if

          g4= g4 + t*(gpmr(1,1,0)*glr(1,2,-1,mmax-2) -
     &                gpmr(1,2,0)*glr(2,2,1,mmax-2)  +
     &                dconjg(glr(1,1,mmax-3,0))*gpmr(1,2,mmax-2) -
     &                dconjg(glr(2,1,mmax-1,0))*gpmr(2,2,mmax-2))
c
          do 40 k=-n,n
           g1=g1+t2*(dconjg(glr(1,1,k,0))*gpmr(1,1,k+1)*glr(1,1,k,mmax)+
     &               dconjg(glr(2,1,k,0))*gpmr(2,2,k-1)*glr(2,1,k,mmax))
     &      + t2*(dconjg(gr(1,1,k,0))*gpml(1,1,k-1)*gr(1,1,k,mmax)+
     &            dconjg(gr(2,1,k,0))*gpml(2,2,k+1)*gr(2,1,k,mmax))    
     &      - t2*(dconjg(glr(1,1,k,0))*gpmr(1,2,k+1)*glr(2,1,k+2,mmax)+
     &            dconjg(glr(2,1,k,0))*gpmr(2,1,k-1)*glr(1,1,k-2,mmax))
     &      - t2*(dconjg(gr(1,1,k,0))*gpml(1,2,k-1)*gr(2,1,k-2,mmax)+
     &            dconjg(gr(2,1,k,0))*gpml(2,1,k+1)*gr(1,1,k+2,mmax))
c
         g3=g3+t2*(dconjg(glr(1,2,k,-2))*gpmr(1,1,k+1)*glr(1,1,k,mmax)+
     &             dconjg(glr(2,2,k,-2))*gpmr(2,2,k-1)*glr(2,1,k,mmax))
     &     + t2*(dconjg(gr(1,2,k,-2))*gpml(1,1,k-1)*gr(1,1,k,mmax)+
     &           dconjg(gr(2,2,k,-2))*gpml(2,2,k+1)*gr(2,1,k,mmax))
     &     - t2*(dconjg(glr(1,2,k,-2))*gpmr(1,2,k+1)*glr(2,1,k+2,mmax)+
     &           dconjg(glr(2,2,k,-2))*gpmr(2,1,k-1)*glr(1,1,k-2,mmax))
     &     - t2*(dconjg(gr(1,2,k,-2))*gpml(1,2,k-1)*gr(2,1,k-2,mmax)+
     &           dconjg(gr(2,2,k,-2))*gpml(2,1,k+1)*gr(1,1,k+2,mmax))
c
         g4=g4+t2*(dconjg(glr(1,1,k,0))*gpmr(1,1,k+1)*glr(1,2,k,mmax-2)+
     &             dconjg(glr(2,1,k,0))*gpmr(2,2,k-1)*glr(2,2,k,mmax-2))
     &   + t2*(dconjg(gr(1,1,k,0))*gpml(1,1,k-1)*gr(1,2,k,mmax-2)+
     &         dconjg(gr(2,1,k,0))*gpml(2,2,k+1)*gr(2,2,k,mmax-2))
     &   - t2*(dconjg(glr(1,1,k,0))*gpmr(1,2,k+1)*glr(2,2,k+2,mmax-2)+
     &         dconjg(glr(2,1,k,0))*gpmr(2,1,k-1)*glr(1,2,k-2,mmax-2))
     &   - t2*(dconjg(gr(1,1,k,0))*gpml(1,2,k-1)*gr(2,2,k-2,mmax-2)+
     &         dconjg(gr(2,1,k,0))*gpml(2,1,k+1)*gr(1,2,k+2,mmax-2))
 40       continue  
c
c Current Density
c
          curr=t2*(dconjg(gr(1,1,mmax,0))*gpml(1,1,mmax-1) -
     &             gpml(1,1,-1)*gr(1,1,0,mmax) +
     &             g1*(g0l(1,1,mmax-1)-dconjg(g0l(1,1,-1)))) +
     &         t2*(gpml(1,2,-1)*gr(2,1,-2,mmax) -
     &             dconjg(gr(2,1,mmax-2,0))*gpml(2,1,mmax-1) +
     &             dconjg(g0l(2,1,-1))*g3 - g4*g0l(2,1,mmax-1))

          zintegrand = curr
c          write(*,*)w,v,curr
c          stop

        return
        end
*********************************************************************
c SUBROUTINE GAP
******************
c
c This subroutine calculates the gap parameter at finite temperature
c within the BCS theory.
c
    
      subroutine gap(temp,gapt)
      implicit real*8 (a-h,o-z)
      common/spar/gamma,pi
      external fct

      np=10000
      eps=1.d-5
      gamma=0.5772156649
      pi=4.d0*datan(1.d0)
      gapt=1.d0
c tempf = temp/tempcritical
      tempf=temp*pi/dexp(gamma)
c
c WARNINGS
c
      if(tempf.gt.1.d0) then
         write(*,*)'# TEMPERATURE IS GREATER THAN THE CRITICAL
     +               TEMPERATURE!'
         gapt=0.d0
         return
      end if

      if(tempf.eq.0.d0) then
         gapt=1.d0
         return
      end if

      if (tempf.gt.0.99d0) then
          gapt=dexp(gamma)*dsqrt((8.d0*(1.d0-tempf))/(7.d0*1.202d0))
          return
      end if
      
 10   xw=(50.d00*tempf)/(gapt*pi*dexp(-gamma))
      dw=xw/dfloat(np)
      a=0.d0
      do 20 j=0,np
         x=dfloat(j)*dw
         dww=dw
         if(j.eq.0.or.j.eq.np) dww=dw/2.0
         a=a+fct(x,gapt,tempf)*dww
 20   continue
      gaptf=dexp(a)
      if (abs(gaptf-gapt).lt.eps) then
          gapt=gaptf
      else
          gapt=gaptf
          go to 10
      end if 
      end
C **************************************************************
      function fct(x,gapt,tempf)
      implicit real*8 (a-h,o-z)
      common/spar/gamma,pi

      w=gapt*dsqrt(x*x + 1.d0)*pi*dexp(-gamma)
      if ((w/tempf).lt.50.d0) then
         fermi=1.d0/(1.d0 + dexp(w/tempf))
      else
         fermi=0.d0
      end if
      fct=-2.d0*fermi/dsqrt(x*x + 1.d0)
      return
      end
C **************************************************************
c------------------------------------------------------------------------
c
      complex*16 function zint(llim,ulim,Atol,Rtol,ierr)
c
c------------------------------------------------------------------------
c
      implicit none
      integer ierr
      complex*16 Atol, Rtol
      real*8 llim, ulim
c
      integer stksz
      parameter (stksz = 4096)
      real*8 xstk(stksz)
      complex*16 zstk(stksz)
c
      integer itask, istk
      complex*16 lz, mz, uz, zres, zintegrand
      real*8 lx, mx, ux
      real*8  zero, two
      parameter (zero=0.0d0, two=2.0d0)
c
c------------------------------------------------------------------------
c
c   **Set some things to zero.**
      istk = 0
      ierr = 0
      zint = dcmplx(zero,zero)
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
      parameter (stksz = 4096)

      integer istk, itask
      complex*16 zres, lz, mz, uz, Atol, Rtol, zstk(stksz)
      real*8 lx, mx, ux, xstk(stksz)
c
      complex*16 ctest, lmz, umz, zintegrand
      real*8 dx, lmx, umx, tmpr, tmpi
      real*8 two, three, four
      parameter (two=2.d0, three=3.d0, four=4.d0)
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
      if ( dabs(dreal(zres-ctest)) .lt. dreal(Atol) ) then
        if ( dabs(dimag(zres-ctest)) .lt. dimag(Atol) ) then
          itask = 0
        else
          tmpi = two*dabs(dimag(zres-ctest)) / dabs(dimag(zres+ctest))
          if ( tmpi .lt. dimag(Rtol) ) then
            itask = 0
          else
            itask = 1
          end if
        end if
      else
        tmpr = two*dabs(dreal(zres-ctest))  / dabs(dreal(zres+ctest))
        if ( tmpr .lt. dreal(Rtol) ) then
          if ( dabs(dimag(zres-ctest)) .lt. dimag(Atol) ) then
            itask = 0
          else
            tmpi = two*dabs(dimag(zres-ctest)) / dabs(dimag(zres+ctest))
            if ( tmpi .lt. dimag(Rtol) ) then
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
