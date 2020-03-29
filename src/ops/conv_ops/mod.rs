use crate::ndarray_ext;
use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::same_type;
use crate::uninitialized_vec;
use crate::Float;
use crate::Tensor;
use ndarray;
#[allow(unused_imports)]
use rayon::iter::*;
use std::f32;
use std::slice;

#[macro_use]
pub mod conv2d;
#[macro_use]
pub mod conv2d_transpose;
pub mod max_pool2d;
#[cfg(feature = "mkl")]
use crate::ndarray_ext::{get_batch_ptrs, get_batch_ptrs_mut};
#[cfg(feature = "mkl")]
use crate::ops::mkl_ffi::{
    cblas_dgemm, cblas_dgemm_batch, cblas_sgemm, cblas_sgemm_batch, CblasTranspose::CblasNoTrans,
    CblasTranspose::CblasTrans, MklInt, CBLAS_ROW_MAJOR,
};

#[test]
fn test_im2col_batch() {
    let op = conv2d::Conv2D {
        pad: 0,
        stride: 1,
        dilation: 1,
    };

    let xch = 2;
    let (xh, xw) = (3, 3);
    let (kh, kw) = (2, 2);

    let x: Vec<f32> = vec![(0..xch * xw * xh).map(|a| a as f32).collect::<Vec<f32>>(); 2]
        .into_iter()
        .flat_map(|a| a)
        .collect();

    let batch_size = 2;

    let ret = im2col_batch(
        x.as_slice(),
        batch_size,
        xch as i32,
        xh as i32,
        xw as i32,
        kh as i32,
        kw as i32,
        op.pad as i32,
        op.pad as i32,
        op.stride as i32,
        op.stride as i32,
        op.dilation as i32,
        op.dilation as i32,
    );

    assert_eq!(
        ret,
        vec![
            0.0, 1.0, 3.0, 4.0, 1.0, 2.0, 4.0, 5.0, 3.0, 4.0, 6.0, 7.0, 4.0, 5.0, 7.0, 8.0, 9.0,
            10.0, 12.0, 13.0, 10.0, 11.0, 13.0, 14.0, 12.0, 13.0, 15.0, 16.0, 13.0, 14.0, 16.0,
            17.0, 0.0, 1.0, 3.0, 4.0, 1.0, 2.0, 4.0, 5.0, 3.0, 4.0, 6.0, 7.0, 4.0, 5.0, 7.0, 8.0,
            9.0, 10.0, 12.0, 13.0, 10.0, 11.0, 13.0, 14.0, 12.0, 13.0, 15.0, 16.0, 13.0, 14.0,
            16.0, 17.0,
        ]
    )
}

#[allow(unused_mut)]
fn im2col_batch<T: Float>(
    x: &[T],           // 4-dimensional
    batch_size: usize, // x.shape[0]
    xch: i32,          // number of channels of x
    xh: i32,
    xw: i32, // x (input) height, width
    kh: i32,
    kw: i32, // kernel height, width
    ph: i32,
    pw: i32, // padding height, width
    sh: i32,
    sw: i32, // stride height, width
    dh: i32,
    dw: i32, // dilation height, width
) -> Vec<T> {
    use std::ptr;

    let yh = (xh + 2 * ph - (dh * (kh - 1) + 1)) / sh + 1;
    let yw = (xw + 2 * pw - (dw * (kw - 1) + 1)) / sw + 1;
    let channel_size = (xh * xw) as usize;
    let size_per_batch_y = (xch * kw * kh * yh * yw) as usize;

    unsafe {
        let mut ret = uninitialized_vec::<T>(batch_size * size_per_batch_y);
        // parallelize outer loop
        (0..batch_size).into_par_iter().for_each(|i| {
            let mut x: *const T = x.get_unchecked(i * xch as usize * channel_size) as *const _;
            let mut ret = ret.get_unchecked(i * size_per_batch_y) as *const T as *mut T;
            for _ in 0..xch {
                for cur_kh in 0..kh {
                    let y_start: i32 = cur_kh * dh - ph;
                    for cur_kw in 0..kw {
                        let x_start = cur_kw * dw - ph;
                        let mut y_offset = y_start;
                        for _ in 0..yh {
                            if (y_offset as u32) < (xh as u32) {
                                let mut x_offset = x_start;
                                let cache = y_offset * xw;
                                for j in 0..yw {
                                    if (x_offset as u32) < (xw as u32) {
                                        *ret.offset(j as isize) =
                                            *x.offset((cache + x_offset) as isize);
                                    } else {
                                        *ret.offset(j as isize) = T::zero();
                                    }
                                    x_offset += sw;
                                }
                            } else {
                                ptr::write_bytes(ret, 0, yw as usize);
                            }
                            ret = ret.offset(yw as isize);
                            y_offset += sh;
                        }
                    }
                }
                x = x.add(channel_size);
            }
        });
        ret
    }
}

fn col2im_batch<T: Float>(
    x: &[T],           // 6-dimensional cols
    batch_size: usize, // x.shape[0]
    xch: i32,          // number of channels of x
    xh: i32,
    xw: i32, // x (input) height, width
    kh: i32,
    kw: i32, // kernel height, width
    ph: i32,
    pw: i32, // padding height, width
    sh: i32,
    sw: i32, // stride height, width
    dh: i32,
    dw: i32, // dilation height, width
) -> Vec<T> {
    let yh = (xh + 2 * ph - (dh * (kh - 1) + 1)) / sh + 1;
    let yw = (xw + 2 * pw - (dw * (kw - 1) + 1)) / sw + 1;
    let channel_size = xh * xw;
    let size_per_batch_x = xch * kh * kw * yh * yw;

    // 4-dimensional
    let ret = vec![T::zero(); batch_size * (xch * xh * xw) as usize];

    // parallelize outer loop
    (0..batch_size).into_par_iter().for_each(|i| unsafe {
        let mut x: *const T = x.get_unchecked(i * size_per_batch_x as usize) as *const T;
        let mut ret = ret.get_unchecked(i * (xch * xh * xw) as usize) as *const T as *mut T;
        for _ in 0..xch {
            for ky in 0..kh {
                let y_start = ky * dh - ph;
                for kx in 0..kw {
                    let x_start = kx * dw - pw;
                    let mut y_offset = y_start;
                    for _ in 0..yh {
                        if (y_offset as u32) < (xh as u32) {
                            let mut x_offset = x_start;
                            let cache = y_offset * xw;
                            for j in 0..yw as isize {
                                if (x_offset as u32) < (xw as u32) {
                                    *ret.add((cache + x_offset) as usize) += *x.offset(j);
                                }
                                x_offset += sw;
                            }
                        }
                        x = x.add(yw as usize);
                        y_offset += sh;
                    }
                }
            }
            ret = ret.add(channel_size as usize);
        }
    });
    ret
}
