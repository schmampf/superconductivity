c                      PROGRAM MAR-FCS

c   ***************************************************************
c   This program computes the FCS of an asymmetric S-S junction
c   ***************************************************************

c
c VARIABLE DEFINITION
c
       implicit real*8 (a-h,o-z)
       parameter (ns=2000)
       integer ierr,nmax,nchi
       complex*16 ui,cvec(3)
       complex*16 p(-ns:ns)
       real*8 curr(-ns:ns)

       common/param/v,trans,temp,dmvt1,dmvt2,eta1,eta2
       common/ndimen/nmax,nchi
      
c Reading the input parameters 

       read(5,*)trans
       read(5,*)temp
       read(5,*)delta1,delta2
       read(5,*)eta1,eta2
       read(5,*)vi,vf,dv
       read(5,*)nmax,iw,nchi

c BEGINNING OF THE PROGRAM SENTENCES
c **********************************
c
c Some constants and parameters
       pi = 4.0d0*datan(1.0d0)
       ui = (0.d0,1.0d0)
c Number of voltage points
       iv = int((vf-vi)/dv)
c Flags
       if(temp.eq.0.d0) temp = 1.d-7
       if(eta1.eq.0.d0) eta1 = 1.d-6
       if(eta2.eq.0.d0) eta2 = 1.d-6

c Subroutine gap calculates the gap at finite temperature.
       if (delta1.gt.0.d0) then
           call gap(0.08617d0*temp/delta1,dmvt1)
           dmvt1 = dmvt1*delta1
       else
           dmvt1 = 0.d0
       end if
       if (delta2.gt.0.d0) then
           call gap(0.08617d0*temp/delta2,dmvt2)
           dmvt2 = dmvt2*delta2
       else
           dmvt2 = 0.d0
       end if
c       write(*,*)'#','delta1 = ',dmvt1,' delta2 = ',dmvt2
c Temperature in meV
       temp = 0.08617d0*temp

c Voltage loop

       DO k=0,iv
          v = vi + dfloat(k)*dv
          wi = -v
          wf = 0.d0
          de = (wf-wi)/dfloat(iw)

          current = 0.d0
          snoise = 0.d0
          c3 = 0.d0
          do j=-2*nmax-2,2*nmax+2
             curr(j) = 0.d0
          end do
	  do i=0,iw
	     w = wi + dfloat(i)*de
             dee = de
	     call green(cvec,w,3,p)
             if(i.eq.0.or.i.eq.iw) dee = de/2.d0
             current = current + dreal(cvec(1))*dee
             snoise = snoise + dreal(cvec(2))*dee
             c3 = c3 + dreal(cvec(3))*dee
             do j=-2*nmax-2,2*nmax+2
                curr(j) = curr(j) + dabs(dfloat(j))*dreal(p(j))*dee
             end do
          end do
          
c Voltage in mV and current in nA
          do j=-2*nmax-2,2*nmax+2
             curr(j) = curr(j)*77.4809d0
          end do
          current = current*77.4809d0

        write(*,200)v,current,curr(-1)-curr(1),
     &  curr(-2)-curr(2),curr(-3)-curr(3),curr(-4)-curr(4),
     &  curr(-5)-curr(5),curr(-6)-curr(6),curr(-7)-curr(7),
     &  curr(-8)-curr(8),curr(-9)-curr(9),curr(-10)-curr(10)

       END DO

 100   format(1x,24f15.10)
 200   format(1x,12f20.15)

       END

*************************************************************************
c
c SUBROUTINE GREEN (calculation of the probabilities, current and noise)
c
******************
        subroutine green(cvec,w,ndim,p)
        implicit real*8 (a-h,o-z)
        parameter (ns=2000)
        integer nan,ndim
	real*8 w,f(-ns:ns),ap(-ns:ns)
        complex*16 ui,wwj1,wwj2,omega1,omega2,ep,em,cvec(ndim),
     &          g1r(-ns:ns),f1r(-ns:ns),g1a(-ns:ns),f1a(-ns:ns),
     &          g2r(-ns:ns),f2r(-ns:ns),g2a(-ns:ns),f2a(-ns:ns),
     &          a(4,4,-ns:ns),b(4,4,-ns:ns),c(4,4,-ns:ns),
     &          aux1(4,4),aux2(4,4),det,p(-ns:ns)
        complex*16 work(4**2)
        integer ipiv(4), info

        common/param/v,trans,temp,delta1,delta2,eta1,eta2
        common/ndimen/nan,nchi

        pi=4.0d0*datan(1.0d0)
        ui = (0.d0,1.d0)
        ck = dsqrt(trans)/2.d0
        dw = 2.0d0*pi/dfloat(nchi) 

c Leads Green functions:

        do j=-nan-1,nan+1
           wj = w + dfloat(j)*v
           wwj1 = wj + ui*eta1
           wwj2 = wj + ui*eta2
	   omega1 = cdsqrt(delta1**2 - wwj1**2)
	   omega2 = cdsqrt(delta2**2 - wwj2**2)

           g1r(j) = -ui*wwj1/omega1
           f1r(j) = delta1/omega1
           g1a(j) = -dconjg(g1r(j))
           f1a(j) = dconjg(f1r(j))

           g2r(j) = -ui*wwj2/omega2
           f2r(j) = delta2/omega2
           g2a(j) = -dconjg(g2r(j))
           f2a(j) = dconjg(f2r(j))

           f(j) = 0.5d0*(1.d0 - dtanh(0.5d0*wj/temp))
        end do

c Q-matrix:

           do j=-nan,nan,2

              a(1,1,j) = 1.d0 + ck*((g1a(j+1)-g1r(j+1))*f(j+1) +
     &                   g1r(j+1) - (g2a(j)-g2r(j))*f(j) - g2r(j))
              a(2,2,j) = 1.d0 + ck*(g1a(j+1)-(g1a(j+1)-g1r(j+1))*f(j+1)-
     &                     g2a(j)+(g2a(j)-g2r(j))*f(j))

              a(1,3,j) = -ck*((f2a(j)-f2r(j))*f(j) + f2r(j))
              a(1,4,j) = -ck*(f2a(j)-f2r(j))*f(j)
              a(2,3,j) = -ck*(f2a(j)-f2r(j))*(1.d0-f(j))
              a(2,4,j) = -ck*(f2a(j) - (f2a(j)-f2r(j))*f(j))

              a(3,1,j) = -ck*((f2a(j)-f2r(j))*f(j) + f2r(j))
              a(3,2,j) = -ck*(f2a(j)-f2r(j))*f(j)
              a(4,1,j) = -ck*(f2a(j)-f2r(j))*(1.0-f(j))
              a(4,2,j) = -ck*(f2a(j) - (f2a(j)-f2r(j))*f(j))

              a(3,3,j) = 1.d0 + ck*(-(g1a(j-1)-g1r(j-1))*f(j-1) -
     &                g1r(j-1) + (g2a(j)-g2r(j))*f(j) + g2r(j))
              a(4,4,j) = 1.d0+ck*(-g1a(j-1)+(g1a(j-1)-g1r(j-1))*f(j-1)+
     &                     g2a(j)-(g2a(j)-g2r(j))*f(j))

              b(3,2,j) = ck*(f1a(j-1)-f1r(j-1))*f(j-1) 
              b(4,1,j) = ck*(f1a(j-1)-f1r(j-1))*(1.0-f(j-1)) 

              c(1,4,j) = ck*(f1a(j+1)-f1r(j+1))*f(j+1) 
              c(2,3,j) = ck*(f1a(j+1)-f1r(j+1))*(1.0-f(j+1)) 

           END DO
            
c Chi loop:
        do j=-2*nan-2,2*nan+2
           p(j) = 0.d0
        end do
        DO ic=0,nchi
           chi = dfloat(ic)*dw
	   dww = dw
           ep = cdexp(ui*chi)
           em = cdexp(-ui*chi)

c Q-matrix:

           do j=-nan,nan,2

              a(1,2,j) = ck*(ep*(g1a(j+1)-g1r(j+1))*f(j+1) -
     &                     (g2a(j)-g2r(j))*f(j))
              a(2,1,j) = ck*(em*(g1a(j+1)-g1r(j+1))*(1.0-f(j+1))-
     &                     (g2a(j)-g2r(j))*(1.0-f(j)))

              a(3,4,j) = ck*(-em*(g1a(j-1)-g1r(j-1))*f(j-1) +
     &                     (g2a(j)-g2r(j))*f(j))
              a(4,3,j) = ck*(-ep*(g1a(j-1)-g1r(j-1))*(1.0-f(j-1))+
     &                     (g2a(j)-g2r(j))*(1.0-f(j)))

              b(3,1,j) = ck*em*((f1a(j-1)-f1r(j-1))*f(j-1) + f1r(j-1))
              b(4,2,j) = ck*ep*(-(f1a(j-1)-f1r(j-1))*f(j-1) + f1a(j-1))

              c(1,3,j) = ck*ep*((f1a(j+1)-f1r(j+1))*f(j+1) + f1r(j+1))
              c(2,4,j) = ck*em*(-(f1a(j+1)-f1r(j+1))*f(j+1) + f1a(j+1))

           END DO
            
c Calculation of the determinant of Q:

           do i=1,4
              do j=1,4
                 aux1(i,j) = a(i,j,-nan)
              end do
           end do
           call zgetrf(4,4,aux1,4,IPIV,INFO)
           icont = 1
           do i=1,4
              if(ipiv(i).ne.i) icont=-icont
           end do
           det = (1.d0,0.d0)*dfloat(icont)
	   do i=1,4
	      det = det*aux1(i,i)
           end do
           call zgetri(4,aux1,4,IPIV,WORK,4**2,INFO)

           do j=-nan+2,nan,2
           
              do k=1,4
                 do l=1,4
                    aux2(k,l) = a(k,l,j)
                 end do
              end do
              do k=3,4
                 do l=3,4
                    do i1=1,2
                       do i2=1,2
          aux2(k,l) = aux2(k,l) - b(k,i1,j)*aux1(i1,i2)*c(i2,l,j-2)
                       end do
                    end do
                 end do
              end do
              do k=1,4
                 do l=1,4
                    aux1(k,l) = aux2(k,l)
                 end do
              end do
              call zgetrf(4,4,aux1,4,IPIV,INFO)
              icont = 1
              do i=1,4
                 if(ipiv(i).ne.i) icont=-icont
              end do
              det = det*dfloat(icont)
	      do i=1,4
	         det = det*aux1(i,i)
              end do
              call zgetri(4,aux1,4,IPIV,WORK,4**2,INFO)

           end do

c           write(*,*)chi/(2.d0*pi),dreal(det),dimag(det)

c Determination of the P_n by a Fourier analysis:

           if(ic.eq.0.or.ic.eq.nchi) dww = dw/2.d0
           do j=-2*nan-2,2*nan+2
              p(j) = p(j) + det*cdexp(-ui*dfloat(j)*chi)*dww/(2.d0*pi)
           end do

        END DO
c        stop

        ptot = 0.d0
        do j=-2*nan-2,2*nan+2
           ptot = ptot + dreal(p(j))
        end do

        do j=-2*nan-2,2*nan+2
           p(j) = p(j)/ptot
           if (j.eq.-2*nan-2) then
                pmax = p(j)
           else
                if(j.ne.0.and.dreal(p(j)).gt.pmax) pmax = dreal(p(j))
           end if
        end do

        do j=-2*nan-2,2*nan+2
           if(dreal(p(j)).lt.pmax/1000.d0) p(j) = 0.d0
        end do

c Current and noise:

	cvec(1) = (0.d0,0.d0)
	cvec(2) = (0.d0,0.d0)
	cvec(3) = (0.d0,0.d0)
	do i =-2*nan-2,2*nan+2
	   cvec(1) = cvec(1) - dfloat(i)*p(i)
	   cvec(2) = cvec(2) + (dabs(dfloat(i))**2)*p(i)
	   cvec(3) = cvec(3) - (dfloat(i)**3)*p(i)
        end do
	cvec(2) = cvec(2) - cvec(1)**2
        cvec(3) = cvec(3) - 3.d0*cvec(1)*cvec(2) - cvec(1)**3

 100    format(1x,9f14.10)
	return
	end

c ***************************************************************
c SUBROUTINE GAP
c ***************************************************************
c
c This subroutine calculates the gap parameter at finite temperature
c within the BCS theory.
c
    
      subroutine gap(temp,gapt)
      implicit real*8 (a-h,o-z)
      common/spar/gamma,pi
      external fct

      np=10000
      eps=0.00001d0
      gamma=0.5772156649d0
      pi=4.d0*datan(1.0d0)
      gapt=1.0d0
c tempf = temp/tempcritical
      tempf=temp*pi/dexp(gamma)

c
c WARNINGS
c
      if(tempf.gt.1.0d0) then
         write(*,*)' BEWARE: THE TEMPERATURE IS GREATER THAN THE BCS CRITICAL
     +               TEMPERATURE!'
         gapt=0.0
         return
      end if

      if(tempf.eq.0.0d0) then
         gapt=1.0d0
         return
      end if

      if (tempf.gt.0.990d0) then
          gapt=dexp(gamma)*dsqrt((8.d0*(1.0d0-tempf))/(7.0d0*1.202d0))
          return
      end if
      
 10   xw=(50.0d0*tempf)/(gapt*pi*dexp(-gamma))
      dw=xw/dfloat(np)
      a=0.d0
      do 20 j=0,np
         x=dfloat(j)*dw
         dww=dw
         if(j.eq.0.or.j.eq.np) dww=dw/2.0d0
         a=a+fct(x,gapt,tempf)*dww
 20   continue
      gaptf=dexp(a)
      if (dabs(gaptf-gapt).lt.eps) then
          gapt=gaptf
      else
          gapt=gaptf
          go to 10
      end if 
      end
c ***************************************************************
      function fct(x,gapt,tempf)
      implicit real*8 (a-h,o-z)
      common/spar/gamma,pi

      w=gapt*dsqrt(x*x + 1.0d0)*pi*dexp(-gamma)
      if ((w/tempf).lt.50.0d0) then
         fermi=1.0d0/(1.0d0 + dexp(w/tempf))
      else
         fermi=0.0d0
      end if
      fct=-2.0*fermi/dsqrt(x*x + 1.0d0)
      return
      end
c ***************************************************************
