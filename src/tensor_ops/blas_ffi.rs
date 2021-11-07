#[cfg(feature = "blas")]
pub(crate) use crate::cblas_sys::*;

// #[cfg(all(feature = "blas", feature = "intel-mkl"))]
// pub(crate) use crate::intel_mkl_sys::*;

#[cfg(feature = "blas")]
#[allow(dead_code)]
pub(crate) enum MemoryOrder {
    C,
    F,
}

#[cfg(feature = "blas")]
pub(crate) type BlasIF = i32;

#[cfg(all(feature = "blas", feature = "intel-mkl"))]
extern "C" {
    pub(crate) fn vsSin(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdSin(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsAsin(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdAsin(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsSinh(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdSinh(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsAsinh(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdAsinh(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsCos(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdCos(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsAcos(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdAcos(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsCosh(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdCosh(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsAcosh(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdAcosh(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsTan(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdTan(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsAtan(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdAtan(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsTanh(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdTanh(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsAtanh(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdAtanh(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsExp(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdExp(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsExp2(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdExp2(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsExp10(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdExp10(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsLn(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdLn(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsLog2(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdLog2(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsLog10(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdLog10(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsInv(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdInv(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsDiv(
        n: BlasIF,
        a: *const libc::c_float,
        b: *const libc::c_float,
        y: *mut libc::c_float,
    );
    pub(crate) fn vdDiv(
        n: BlasIF,
        a: *const libc::c_double,
        b: *const libc::c_double,
        y: *mut libc::c_double,
    );

    pub(crate) fn vsSqrt(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdSqrt(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsPowx(
        n: BlasIF,
        a: *const libc::c_float,
        b: libc::c_float,
        y: *mut libc::c_float,
    );
    pub(crate) fn vdPowx(
        n: BlasIF,
        a: *const libc::c_double,
        b: libc::c_double,
        y: *mut libc::c_double,
    );

    pub(crate) fn vsInvSqrt(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdInvSqrt(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsAdd(
        n: BlasIF,
        a: *const libc::c_float,
        b: *const libc::c_float,
        y: *mut libc::c_float,
    );
    pub(crate) fn vdAdd(
        n: BlasIF,
        a: *const libc::c_double,
        b: *const libc::c_double,
        y: *mut libc::c_double,
    );

    pub(crate) fn vsSub(
        n: BlasIF,
        a: *const libc::c_float,
        b: *const libc::c_float,
        y: *mut libc::c_float,
    );
    pub(crate) fn vdSub(
        n: BlasIF,
        a: *const libc::c_double,
        b: *const libc::c_double,
        y: *mut libc::c_double,
    );

    pub(crate) fn vsSqr(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdSqr(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsMul(
        n: BlasIF,
        a: *const libc::c_float,
        b: *const libc::c_float,
        y: *mut libc::c_float,
    );
    pub(crate) fn vdMul(
        n: BlasIF,
        a: *const libc::c_double,
        b: *const libc::c_double,
        y: *mut libc::c_double,
    );

    pub(crate) fn vsAbs(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdAbs(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsFloor(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdFloor(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsCeil(n: BlasIF, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdCeil(n: BlasIF, a: *const libc::c_double, y: *mut libc::c_double);
}
