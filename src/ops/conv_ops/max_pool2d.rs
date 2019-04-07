use super::*;
use tensor::Tensor;

pub struct MaxPool2D {
    pub pad: usize,
    pub stride: usize,
    pub size: usize,
}

pub struct MaxPool2DGrad {
    pad: usize,
    stride: usize,
    size: usize,
}

pub struct MaxPool2DGradGrad {
    pad: usize,
    stride: usize,
    size: usize,
}

macro_rules! impl_max_pool {
    ($t:ty, $i:ident) => {
        unsafe fn $i<T: Float>(
            input: *const T,
            pad: usize,
            xh: usize,
            xw: usize,
            yh: usize,
            yw: usize,
            ch: usize,
            batch: usize,
            size: usize,
            stride: usize,
        ) -> (Vec<T>, Vec<T>) {
            let all_len_y = batch * ch * yh * yw;
            let mut indices = uninitialized_vec(all_len_y);
            let mut output = uninitialized_vec(all_len_y);
            for b in 0..batch {
                for c in 0..ch {
                    let c_base = xh * (c + b * ch);
                    for i in 0..yh {
                        let i_base = yw * (i + yh * (c + b * ch));
                        let mut h_start = i * stride - pad;
                        let h_end = if h_start + size > xh {
                            xh
                        } else {
                            h_start + size
                        };
                        h_start = h_start * (h_start > 0) as usize;
                        for j in 0..yw {
                            let mut max = T::min_value();
                            let mut max_i = 0; // default
                            let mut w_start = j * stride - pad;
                            let w_end = if w_start + size > xw {
                                xw
                            } else {
                                w_start + size
                            };
                            w_start = w_start * (w_start > 0) as usize;
                            // in a window
                            for h in h_start..h_end {
                                let rows = xw * (h + c_base);
                                for w in w_start..w_end {
                                    let index = w + rows;
                                    let val = *input.offset(index as isize);
                                    if val > max {
                                        max_i = index;
                                        max = val;
                                    }
                                }
                            }
                            let out_index = j + i_base;
                            *output.get_unchecked_mut(out_index) = max;
                            *indices.get_unchecked_mut(out_index) = *(&(max_i as $t) as *const $t as *const T)
                        }
                    }
                }
            }
            (output, indices)
        }
    };
}

impl_max_pool!(f32, max_pool_f32);
impl_max_pool!(f64, max_pool_f64);

#[test]
fn test_max_pool() {
    let x = vec![0., 1., 2., 5., 4., 3., 6., 7., 8.];
    let (output, argmax) = unsafe {
        max_pool_f64(
            x.as_ptr(),
            0, // pad
            3,
            3, // h, w
            2,
            2, // out_h, out_w
            1, // c
            1, // batch
            2, // size
            1, // stride
        )
    };
    assert_eq!(output, vec![5., 4., 7., 8.]);
    assert_eq!(argmax, vec![3., 4., 7., 8.]);
}

macro_rules! impl_max_pool_grad {
    ($t:ty, $i:ident) => {
        fn $i<T: Float>(
            batch: usize,
            mut gy: *const T,
            xh: usize,
            xw: usize,
            yh: usize,
            yw: usize,
            c: usize,
            mut argmax: *const $t,
        ) -> Vec<T> {
            let mut ret = vec![T::zero(); batch * c * xh * xw];
            let gx = ret.as_mut_ptr();
            let until = yh * yw * c * batch;
            for _ in 0..until {
                unsafe {
                    *gx.offset(*argmax as isize) += *gy;
                    argmax = argmax.offset(1);
                    gy = gy.offset(1);
                }
            }
            ret
        }
    };
}

macro_rules! impl_max_pool_grad_grad {
    ($t:ty, $i:ident) => {
        fn $i<T: Float>(
            ggx: *const T,
            yh: usize,
            yw: usize,
            c: usize,
            batch: usize,
            mut argmax: *const $t,
        ) -> Vec<T> {
            let mut ret = uninitialized_vec(batch * c * yh * yw);
            let mut ggy = ret.as_mut_ptr();
            let until = yh * yw * c * batch;
            for _ in 0..until {
                unsafe {
                    *ggy = *ggx.offset(*argmax as isize);
                    ggy = ggy.offset(1);
                    argmax = argmax.offset(1);
                }
            }
            ret
        }
    };
}

impl_max_pool_grad!(f32, max_pool_grad_f32);
impl_max_pool_grad!(f64, max_pool_grad_f64);
impl_max_pool_grad_grad!(f32, max_pool_grad_grad_f32);
impl_max_pool_grad_grad!(f64, max_pool_grad_grad_f64);

impl<T: Float> ::op::Op<T> for MaxPool2D {
    fn name(&self) -> &str {
        "MaxPool"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> ::op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x: &NdArray<T> = xs[0];
        let x_shape = x.shape();
        let batch = x_shape[0];
        let c = x_shape[1];
        let xh = x_shape[2];
        let xw = x_shape[3];

        let yh = (xh + 2 * self.pad - self.size) / self.stride + 1;
        let yw = (xw + 2 * self.pad - self.size) / self.stride + 1;
        let (output, indices) = unsafe {
            if same_type::<T, f32>() {
                max_pool_f32(
                    x.as_ptr(),
                    self.pad,
                    xh,
                    xw,
                    yh,
                    yw,
                    c,
                    batch,
                    self.size,
                    self.stride,
                )
            } else if same_type::<T, f64>() {
                max_pool_f64(
                    x.as_ptr(),
                    self.pad,
                    xh,
                    xw,
                    yh,
                    yw,
                    c,
                    batch,
                    self.size,
                    self.stride,
                )
            } else {
                panic!("MaxPoolGrad supports only f32 and f64");
            }
        };
        let output = NdArray::from_shape_vec(ndarray::IxDyn(&[batch, c, yh, yw]), output);
        let indices = NdArray::from_shape_vec(ndarray::IxDyn(&[batch, c, yh, yw]), indices);
        vec![Ok(output.unwrap()), Ok(indices.unwrap())]
    }

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], y: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let indices = ::ops::nth_tensor(y, 1);
        let gx = Tensor::builder()
            .set_inputs(vec![&gy, &indices])
            .build(MaxPool2DGrad {
                pad: self.pad,
                stride: self.stride,
                size: self.size,
            });
        vec![Some(gx)]
    }
}

#[test]
fn test_max_pool2d() {
    use op::Op;

    let op = MaxPool2D {
        pad: 0,
        stride: 1,
        size: 2,
    };
    let x = vec![0., 1., 2., 5., 4., 3., 6., 7., 8.];
    let y = op.compute(::runtime::OpComputeContext::new(
        &::zeros(&[0]),
        vec![&NdArray::from_shape_vec(ndarray::IxDyn(&[1, 1, 3, 3]), x).unwrap()],
    ));
    assert_eq!(
        vec![5., 4., 7., 8.],
        y[0].as_ref().unwrap().as_slice().unwrap()
    );
    assert_eq!(
        vec![3., 4., 7., 8.],
        y[1].as_ref().unwrap().as_slice().unwrap()
    );
}

impl<T: Float> ::op::Op<T> for MaxPool2DGrad {
    fn name(&self) -> &str {
        "MaxPoolGrad"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> ::op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let gy = xs[0];
        let argmax = xs[1];
        let gy_shape = gy.shape();
        let batch = gy_shape[0];
        let c = gy_shape[1];
        let yh = gy_shape[2];
        let yw = gy_shape[3];

        let xh = self.stride * (yh - 1) - 2 * self.pad + self.size;
        let xw = self.stride * (yw - 1) - 2 * self.pad + self.size;
        let gx = if same_type::<T, f32>() {
            max_pool_grad_f32(
                batch,
                gy.as_ptr(),
                xh,
                xw,
                yh,
                yw,
                c,
                argmax.as_ptr() as *const f32,
            )
        } else if same_type::<T, f64>() {
            max_pool_grad_f64(
                batch,
                gy.as_ptr(),
                xh,
                xw,
                yh,
                yw,
                c,
                argmax.as_ptr() as *const f64,
            )
        } else {
            panic!("MaxPoolGrad supports only f32 and f64");
        };
        let gx = NdArray::from_shape_vec(ndarray::IxDyn(&[batch, c, xh, xw]), gx);
        vec![Ok(gx.unwrap())]
    }

    fn grad(&self, ggx: &Tensor<T>, xs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let argmax = xs[1];
        let ggy = Tensor::builder()
            .set_inputs(vec![ggx, argmax])
            .build(MaxPool2DGradGrad {
                pad: self.pad,
                stride: self.stride,
                size: self.size,
            });
        vec![Some(ggy), None]
    }
}

impl<T: Float> ::op::Op<T> for MaxPool2DGradGrad {
    fn name(&self) -> &str {
        "MaxPoolGradGrad"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> ::op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let ggx = xs[0];
        let x_shape = ggx.shape();
        let batch = x_shape[0];
        let c = x_shape[1];
        let xh = x_shape[2];
        let xw = x_shape[3];
        let yh = (xh + 2 * self.pad - self.size) / self.stride + 1;
        let yw = (xw + 2 * self.pad - self.size) / self.stride + 1;
        let argmax = xs[1];
        let ggy = if same_type::<T, f32>() {
            max_pool_grad_grad_f32(
                ggx.as_ptr(),
                yh,
                yw,
                c,
                batch,
                argmax.as_ptr() as *const f32,
            )
        } else if same_type::<T, f64>() {
            max_pool_grad_grad_f64(
                ggx.as_ptr(),
                yh,
                yw,
                c,
                batch,
                argmax.as_ptr() as *const f64,
            )
        } else {
            panic!("MaxPoolGradGrad supports only f32 and f64");
        };
        let ggy = NdArray::from_shape_vec(ndarray::IxDyn(&[batch, c, yh, yw]), ggy).unwrap();
        vec![Ok(ggy)]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None, None]
    }
}
