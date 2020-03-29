#[cfg(feature = "mkl")]
pub(crate) type MklInt = i64;

#[cfg(feature = "mkl")]
#[allow(dead_code)]
pub(crate) enum MemoryOrder {
    C,
    F,
}

#[cfg(feature = "mkl")]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) enum CblasTranspose {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113,
}

#[cfg(feature = "mkl")]
pub(crate) type CblasLayout = usize;

#[cfg(feature = "mkl")]
pub(crate) const CBLAS_ROW_MAJOR: usize = 101;

#[cfg(feature = "mkl")]
extern "C" {
    pub(crate) fn cblas_sgemm(
        layout: CblasLayout,
        transa: CblasTranspose,
        transb: CblasTranspose,
        m: MklInt,
        n: MklInt,
        k: MklInt,
        alpha: libc::c_float,
        a: *const libc::c_float,
        lda: MklInt,
        b: *const libc::c_float,
        ldb: MklInt,
        beta: libc::c_float,
        c: *mut libc::c_float,
        ldc: MklInt,
    );

    pub(crate) fn cblas_dgemm(
        layout: CblasLayout,
        transa: CblasTranspose,
        transb: CblasTranspose,
        m: MklInt,
        n: MklInt,
        k: MklInt,
        alpha: libc::c_double,
        a: *const libc::c_double,
        lda: MklInt,
        b: *const libc::c_double,
        ldb: MklInt,
        beta: libc::c_double,
        c: *mut libc::c_double,
        ldc: MklInt,
    );

    pub(crate) fn cblas_sgemm_batch(
        layout: CblasLayout,
        transa_array: *const CblasTranspose, // batch of CblasTranspose
        transb_array: *const CblasTranspose, // batch of CblasTranspose
        m_array: *const MklInt,              // batch of m
        n_array: *const MklInt,              // batch of n
        k_array: *const MklInt,              // batch of k
        alpha_array: *const libc::c_float,   // batch of alpha
        a_array: *const *const libc::c_float, // a
        lda_array: *const MklInt,            // batch of lda
        b_array: *const *const libc::c_float, // b
        ldb_array: *const MklInt,            // batch of ldb
        beta_array: *const libc::c_float,    // batch of beta
        c_array: *mut *mut libc::c_float,    // c
        ldc_array: *const MklInt,            // batch of odc
        group_count: MklInt,                 // batch size
        group_size: *const MklInt,
    );

    pub(crate) fn cblas_dgemm_batch(
        layout: CblasLayout,
        transa_array: *const CblasTranspose, // batch of CblasTranspose
        transb_array: *const CblasTranspose, // batch of CblasTranspose
        m_array: *const MklInt,              // batch of m
        n_array: *const MklInt,              // batch of n
        k_array: *const MklInt,              // batch of k
        alpha_array: *const libc::c_double,  // batch of alpha
        a_array: *const *const libc::c_double, // a
        lda_array: *const MklInt,            // batch of lda
        b_array: *const *const libc::c_double, // b
        ldb_array: *const MklInt,            // batch of ldb
        beta_array: *const libc::c_double,   // batch of beta
        c_array: *mut *mut libc::c_double,   // c
        ldc_array: *const MklInt,            // batch of odc
        group_count: MklInt,                 // batch size
        group_size: *const MklInt,
    );

    pub(crate) fn vsSin(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdSin(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsAsin(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdAsin(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsSinh(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdSinh(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsAsinh(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdAsinh(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsCos(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdCos(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsAcos(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdAcos(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsCosh(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdCosh(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsAcosh(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdAcosh(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsTan(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdTan(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsAtan(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdAtan(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsTanh(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdTanh(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsAtanh(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdAtanh(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsExp(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdExp(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsExp2(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdExp2(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsExp10(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdExp10(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsLn(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdLn(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsLog2(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdLog2(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsLog10(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdLog10(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsInv(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdInv(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsDiv(
        n: MklInt,
        a: *const libc::c_float,
        b: *const libc::c_float,
        y: *mut libc::c_float,
    );
    pub(crate) fn vdDiv(
        n: MklInt,
        a: *const libc::c_double,
        b: *const libc::c_double,
        y: *mut libc::c_double,
    );

    pub(crate) fn vsSqrt(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdSqrt(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsPowx(
        n: MklInt,
        a: *const libc::c_float,
        b: libc::c_float,
        y: *mut libc::c_float,
    );
    pub(crate) fn vdPowx(
        n: MklInt,
        a: *const libc::c_double,
        b: libc::c_double,
        y: *mut libc::c_double,
    );

    pub(crate) fn vsInvSqrt(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdInvSqrt(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsAdd(
        n: MklInt,
        a: *const libc::c_float,
        b: *const libc::c_float,
        y: *mut libc::c_float,
    );
    pub(crate) fn vdAdd(
        n: MklInt,
        a: *const libc::c_double,
        b: *const libc::c_double,
        y: *mut libc::c_double,
    );

    pub(crate) fn vsSub(
        n: MklInt,
        a: *const libc::c_float,
        b: *const libc::c_float,
        y: *mut libc::c_float,
    );
    pub(crate) fn vdSub(
        n: MklInt,
        a: *const libc::c_double,
        b: *const libc::c_double,
        y: *mut libc::c_double,
    );

    pub(crate) fn vsSqr(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdSqr(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsMul(
        n: MklInt,
        a: *const libc::c_float,
        b: *const libc::c_float,
        y: *mut libc::c_float,
    );
    pub(crate) fn vdMul(
        n: MklInt,
        a: *const libc::c_double,
        b: *const libc::c_double,
        y: *mut libc::c_double,
    );

    pub(crate) fn vsAbs(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdAbs(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsFloor(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdFloor(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);

    pub(crate) fn vsCeil(n: MklInt, a: *const libc::c_float, y: *mut libc::c_float);
    pub(crate) fn vdCeil(n: MklInt, a: *const libc::c_double, y: *mut libc::c_double);
}
