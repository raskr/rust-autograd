use super::*;

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
                                    let val = *input.add(index);
                                    if val > max {
                                        max_i = index;
                                        max = val;
                                    }
                                }
                            }
                            let out_index = j + i_base;
                            *output.get_unchecked_mut(out_index) = max;
                            *indices.get_unchecked_mut(out_index) =
                                *(&(max_i as $t) as *const $t as *const T)
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
        unsafe fn $i<T: Float>(
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
                *ggy = *ggx.offset(*argmax as isize);
                ggy = ggy.offset(1);
                argmax = argmax.offset(1);
            }
            ret
        }
    };
}

impl_max_pool_grad!(f32, max_pool_grad_f32);
impl_max_pool_grad!(f64, max_pool_grad_f64);
impl_max_pool_grad_grad!(f32, max_pool_grad_grad_f32);
impl_max_pool_grad_grad!(f64, max_pool_grad_grad_f64);

impl<T: Float> crate::op::Op<T> for MaxPool2D {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let x = &ctx.input(0);
        let x_shape = x.shape();
        let batch = x_shape[0];
        let c = x_shape[1];
        let xh = x_shape[2];
        let xw = x_shape[3];
        let copied_x = ndarray_ext::copy_if_not_standard(x);
        let x = copied_x.map(|inner| inner.as_ptr()).unwrap_or(x.as_ptr());

        let yh = (xh + 2 * self.pad - self.size) / self.stride + 1;
        let yw = (xw + 2 * self.pad - self.size) / self.stride + 1;
        let (output, indices) = unsafe {
            if same_type::<T, f32>() {
                max_pool_f32(
                    x,
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
                    x,
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
                ctx.append_error(op::OpError::TypeUnsupported(
                    "MaxPool supports only f32 and f64".to_string(),
                ));
                return;
            }
        };
        unsafe {
            let output =
                NdArray::from_shape_vec_unchecked(ndarray::IxDyn(&[batch, c, yh, yw]), output);
            let indices =
                NdArray::from_shape_vec_unchecked(ndarray::IxDyn(&[batch, c, yh, yw]), indices);
            ctx.append_output(output);
            ctx.append_output(indices);
        }
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let s = ctx.graph();
        let gy = &ctx.output_grad();
        let y = &ctx.output();
        let indices = s.nth_tensor(y, 1);
        let gx = Tensor::builder().set_ro_inputs(&[gy, &indices]).build(
            s,
            MaxPool2DGrad {
                pad: self.pad,
                stride: self.stride,
                size: self.size,
            },
        );
        ctx.append_input_grad(Some(gx));
    }
}

impl<T: Float> crate::op::Op<T> for MaxPool2DGrad {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let gy = &ctx.input(0);
        let argmax = &ctx.input(1);
        let gy_shape = gy.shape();
        let batch = gy_shape[0];
        let c = gy_shape[1];
        let yh = gy_shape[2];
        let yw = gy_shape[3];
        let copied_gy = ndarray_ext::copy_if_not_standard(gy);
        let gy = copied_gy.map(|inner| inner.as_ptr()).unwrap_or(gy.as_ptr());

        let xh = self.stride * (yh - 1) - 2 * self.pad + self.size;
        let xw = self.stride * (yw - 1) - 2 * self.pad + self.size;
        let gx = if same_type::<T, f32>() {
            max_pool_grad_f32(batch, gy, xh, xw, yh, yw, c, argmax.as_ptr() as *const f32)
        } else if same_type::<T, f64>() {
            max_pool_grad_f64(batch, gy, xh, xw, yh, yw, c, argmax.as_ptr() as *const f64)
        } else {
            ctx.append_error(op::OpError::TypeUnsupported(
                "MaxPool2DGrad supports only f32 and f64".to_string(),
            ));
            return;
        };
        unsafe {
            let gx = NdArray::from_shape_vec_unchecked(ndarray::IxDyn(&[batch, c, xh, xw]), gx);
            ctx.append_output(gx);
        }
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let ggx = &ctx.output_grad();
        let s = ctx.graph();
        let argmax = &ctx.input(1);
        let ggy = Tensor::builder().set_ro_inputs(&[&ggx, argmax]).build(
            s,
            MaxPool2DGradGrad {
                pad: self.pad,
                stride: self.stride,
                size: self.size,
            },
        );
        ctx.append_input_grad(Some(ggy));
        ctx.append_input_grad(None);
    }
}

impl<T: Float> crate::op::Op<T> for MaxPool2DGradGrad {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ggx = &ctx.input(0);
        let x_shape = ggx.shape();
        let copied_ggx = ndarray_ext::copy_if_not_standard(ggx);
        let ggx = copied_ggx
            .map(|inner| inner.as_ptr())
            .unwrap_or(ggx.as_ptr());
        let batch = x_shape[0];
        let c = x_shape[1];
        let xh = x_shape[2];
        let xw = x_shape[3];
        let yh = (xh + 2 * self.pad - self.size) / self.stride + 1;
        let yw = (xw + 2 * self.pad - self.size) / self.stride + 1;
        let argmax = &ctx.input(1);
        let ggy = unsafe {
            let ggy = if same_type::<T, f32>() {
                max_pool_grad_grad_f32(ggx, yh, yw, c, batch, argmax.as_ptr() as *const f32)
            } else if same_type::<T, f64>() {
                max_pool_grad_grad_f64(ggx, yh, yw, c, batch, argmax.as_ptr() as *const f64)
            } else {
                ctx.append_error(op::OpError::TypeUnsupported(
                    "MaxPool2DGradGrad supports only f32 and f64".to_string(),
                ));
                return;
            };
            NdArray::from_shape_vec_unchecked(ndarray::IxDyn(&[batch, c, yh, yw]), ggy)
        };
        ctx.append_output(ggy);
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None);
        ctx.append_input_grad(None);
    }
}
