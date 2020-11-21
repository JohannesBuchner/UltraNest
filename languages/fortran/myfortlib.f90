    subroutine my_fort_transform(params, d) BIND(C, NAME='my_fort_transform')
        use iso_c_binding
        implicit none
        !DEC$ ATTRIBUTES DLLEXPORT :: my_fort_transform
        integer (C_SIZE_T), intent(in) :: d
        real (C_DOUBLE), dimension(d), intent(inout) :: params
        integer (C_SIZE_T) :: i

        !write(*,*) "Fortran transform called"
        !write(*,*) d

        do i=1,d
          params(i) = params(i) * 2 - 1
        end do
    end subroutine

    subroutine my_fort_likelihood(params, d, l) BIND(C, NAME='my_fort_likelihood')
        use iso_c_binding
        implicit none
        !DEC$ ATTRIBUTES DLLEXPORT :: my_fort_likelihood
        integer (C_INT), intent(in) :: d
        real (C_DOUBLE), dimension(d), intent(inout) :: params
        integer :: i
        real (C_DOUBLE), intent(out) :: l

        l = 0.0
        do i=1,d
           l = l + ((params(i) - (i - 1) * 0.1)/0.01)**2
        end do
        l = -0.5 * l
        !write(*,*) "Fortran likelihood called", l
        !write(*,*) l
    end subroutine
