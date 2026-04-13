c                      PROGRAM MAR-FCS

c   ***************************************************************
c   This program computes the FCS of an asymmetric S-S junction
c   ***************************************************************

c
c VARIABLE DEFINITION
c
c  NOTE (2026-04): the public wrapper entry points have been moved
c  into fcs_api.f90. This file now contains only the numerical kernel
c  used by that API layer.

*************************************************************************
c
c SUBROUTINE GREEN (calculation of the probabilities, current and noise)
c
******************
        subroutine green(cvec,w,ndim,p,v,trans,temp,delta1,delta2,
     &                   eta1,eta2,nan,nchi,
     &                   f,g1r,f1r,g1a,f1a,g2r,f2r,g2a,f2a,
     &                   a,b,c,aux1,aux2,work,ipiv)
        implicit real*8 (a-h,o-z)
        parameter (ns=2000)
        integer nan,ndim,nchi
	real*8 w,v,trans,temp,delta1,delta2,eta1,eta2,f(-ns:ns)
        complex*16 ui,wwj1,wwj2,omega1,omega2,ep,em,cvec(ndim),
     &          g1r(-ns:ns),f1r(-ns:ns),g1a(-ns:ns),f1a(-ns:ns),
     &          g2r(-ns:ns),f2r(-ns:ns),g2a(-ns:ns),f2a(-ns:ns),
     &          a(4,4,-ns:ns),b(4,4,-ns:ns),c(4,4,-ns:ns),
     &          aux1(4,4),aux2(4,4),det,p(-ns:ns)
        complex*16 work(4**2)
        integer ipiv(4), info

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
