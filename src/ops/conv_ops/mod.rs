use ndarray;
use ndarray_ext::NdArray;
#[allow(unused_imports)]
use rayon::iter::*;
use same_type;
use std::f32;
use std::mem;
use std::slice;
use tensor::Tensor;
use Float;

macro_rules! get_xw {
    ($op:expr, $yw:expr, $kw:expr) => {
        $op.stride * ($yw - 1) - 2 * $op.pad + ($op.dilation * ($kw - 1) + 1)
    };
}

macro_rules! get_xh {
    ($op:expr, $yh:expr, $kh:expr) => {
        $op.stride * ($yh - 1) - 2 * $op.pad + ($op.dilation * ($kh - 1) + 1)
    };
}

macro_rules! get_yw {
    ($op:expr, $xw:expr, $kw:expr) => {
        ($xw + 2 * $op.pad - ($op.dilation * ($kw - 1) + 1)) / $op.stride + 1
    };
}

macro_rules! get_yh {
    ($op:expr, $xh:expr, $kh:expr) => {
        ($xh + 2 * $op.pad - ($op.dilation * ($kh - 1) + 1)) / $op.stride + 1
    };
}

#[macro_use]
pub mod conv2d;
#[macro_use]
pub mod conv2d_transpose;
pub mod max_pool2d;

#[inline]
fn uninitialized_vec<T>(size: usize) -> Vec<T> {
    let mut buf = Vec::with_capacity(size);
    unsafe {
        buf.set_len(size);
    }
    buf
}

#[test]
fn test_conv_filter_grad() {
    use op::Op;
    let op = conv2d::Conv2DFilterGrad {
        pad: 0,
        stride: 1,
        dilation: 1,
    };

    let (kh, kw) = (2, 2);
    let (xch, ych) = (3, 2);
    let (yh, yw) = (2, 2);
    let batch_size = 2;

    let x = ::ndarray_ext::ones::<f32>(&[batch_size, yh, yw, kh, kw, xch]);
    let g = ::ndarray_ext::ones(&[batch_size, ych, yh, yw]);
    let w = ::ndarray_ext::ones(&[ych, xch, kh, kw]);

    let ret = op.compute(::runtime::OpComputeContext::new(
        &::ops::zeros(&[0]), // dummy (not used)
        vec![&x, &g, &w],
    ));

    assert_eq!(w.shape(), ret[0].as_ref().unwrap().shape()); // (2, 3, 2, 2)
    assert_eq!(ret[0].clone().unwrap().into_raw_vec(), vec![8.; 24]);
}

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
        xch as isize,
        xh as isize,
        xw as isize,
        kh as isize,
        kw as isize,
        op.pad as isize,
        op.pad as isize,
        op.stride as isize,
        op.stride as isize,
        op.dilation as isize,
        op.dilation as isize,
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

fn im2col_batch<T: Float>(
    x: &[T],           // 4-dimensional
    batch_size: usize, // x.shape[0]
    xch: isize,        // number of channels of x
    xh: isize,
    xw: isize, // x (input) height, width
    kh: isize,
    kw: isize, // kernel height, width
    ph: isize,
    pw: isize, // padding height, width
    sh: isize,
    sw: isize, // stride height, width
    dh: isize,
    dw: isize, // dilation height, width
) -> Vec<T> {
    let yh = (xh + 2 * ph - (dh * (kh - 1) + 1)) / sh + 1;
    let yw = (xw + 2 * pw - (dw * (kw - 1) + 1)) / sw + 1;
    let channel_size = (xh * xw) as usize;
    let size_per_batch_y = (xch * kw * kh * yh * yw) as usize;

    let ret = uninitialized_vec::<T>(batch_size * size_per_batch_y);

    unsafe {
        // parallelize outer loop
        (0..batch_size).into_par_iter().for_each(|i| {
            let mut x: *const T = x.get_unchecked(i * xch as usize * channel_size) as *const _;
            let mut ret: *mut T = mem::transmute(ret.get_unchecked(i * size_per_batch_y));
            for _ in 0..xch {
                for cur_kh in 0..kh {
                    let y_start = cur_kh * dh - ph;
                    let edge1 = if y_start >= 0 {
                        0
                    } else {
                        yh.min((-y_start / sh) + ((y_start % sh) != 0) as isize)
                    };
                    let edge2 = yh.min((xh - y_start) / sh + ((xh - y_start) % sh != 0) as isize);

                    for cur_kw in 0..kw {
                        let x_start = cur_kw * dw - pw;

                        let edge3 = if x_start >= 0 {
                            0
                        } else {
                            yw.min((-x_start / sw) + ((x_start % sw) != 0) as isize)
                        };
                        let edge4 =
                            yw.min((xw - x_start) / sw + ((xw - x_start) % sw != 0) as isize);

                        let mut cur_y = y_start;

                        for _ in 0..edge1 {  // pad
                            for _ in 0..yw {
                                *ret = T::zero();
                                ret = ret.offset(1);
                            }
                            cur_y += sh;
                        }

                        for _ in edge1..edge2 {
                            let mut cur_x = x_start;
                            for _ in 0..edge3 {  // pad
                                *ret = T::zero();
                                ret = ret.offset(1);
                                cur_x += sw;
                            }
                            for _ in edge3..edge4 {
                                *ret = *x.offset((cur_y * xw + cur_x) as isize);
                                ret = ret.offset(1);
                                cur_x += sw;
                            }
                            for _ in edge4..yw {  // pad
                                *ret = T::zero();
                                ret = ret.offset(1);
                                cur_x += sw;
                            }
                            cur_y += sh;
                        }

                        for _ in edge2..yh {  // pad
                            for _ in 0..yw {
                                *ret = T::zero();
                                ret = ret.offset(1);
                            }
                            cur_y += sh;
                        }
                    }
                }
                x = x.offset(channel_size as isize);
            }
        });
    }
    ret
}

fn col2im_batch<T: Float>(
    x: &[T],           // 6-dimensional cols
    batch_size: usize, // x.shape[0]
    xch: isize,        // number of channels of x
    xh: isize,
    xw: isize, // x (input) height, width
    kh: isize,
    kw: isize, // kernel height, width
    ph: isize,
    pw: isize, // padding height, width
    sh: isize,
    sw: isize, // stride height, width
    dh: isize,
    dw: isize, // dilation height, width
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
        let mut ret: *mut T = mem::transmute(ret.get_unchecked(i * (xch * xh * xw) as usize));

        for _ in 0..xch {
            for ky in 0..kh {
                let y_start = ky * dh - ph;
                let edge1 = if y_start >= 0 {
                    0
                } else {
                    yh.min((-y_start / sh) + ((y_start % sh) != 0) as isize)
                };
                let edge2 = yh.min((xh - y_start) / sh + ((xh - y_start) % sh != 0) as isize);

                for kx in 0..kw {
                    let x_start = kx * dw - pw;
                    let edge3 = if x_start >= 0 {
                        0
                    } else {
                        yh.min((-x_start / sw) + ((x_start % sw) != 0) as isize)
                    };
                    let edge4 = yw.min((xw - x_start) / sw + ((xw - x_start) % sw != 0) as isize);
                    let mut cur_y = y_start + sh * edge1;
                    x = x.offset(yw * edge1);

                    for _ in edge1..edge2 {
                        x = x.offset(edge3);
                        let mut cur_x = x_start + sw * edge3;
                        for _ in edge3..edge4 {
                            *ret.offset(cur_y * xw + cur_x) += *x;
                            x = x.offset(1);
                            cur_x += sw;
                        }
                        x = x.offset(yw - edge4);
                        cur_y += sh;
                    }

                    x = x.offset(yw * (yh - edge2));
                }
            }
            ret = ret.offset(channel_size as isize);
        }
    });
    ret
}
