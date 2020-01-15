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
}
