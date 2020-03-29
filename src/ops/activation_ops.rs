use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
#[cfg(feature = "mkl")]
use crate::ops::mkl_ffi::*;
#[cfg(feature = "mkl")]
use crate::same_type;
use crate::tensor::Tensor;
use crate::Float;
use ndarray;
pub struct ELU<T: Float> {
    pub alpha: T,
}

pub struct ELUGrad<T: Float> {
    pub alpha: T,
}

pub struct Identity;

pub struct ReLU;

pub struct Sigmoid;

pub struct Softplus;

pub struct Softmax {
    pub axis: isize,
}

#[cfg(feature = "mkl")]
fn fast_sigmoid_impl<F: Float>(x: &NdArrayView<F>) -> NdArray<F> {
    let half = F::from(0.5).unwrap();
    unsafe {
        if same_type::<F, f32>() {
            let mut y = x.mapv(move |x| x * half);
            vsTanh(
                y.len() as MklInt,
                y.as_ptr() as *const f32,
                y.as_mut_ptr() as *mut f32,
            );
            y.mapv_inplace(move |x2| half * (x2 + F::one()));
            return y;
        } else if same_type::<F, f64>() {
            let mut y = x.mapv(move |x| x * half);
            vdTanh(
                y.len() as MklInt,
                y.as_ptr() as *const f64,
                y.as_mut_ptr() as *mut f64,
            );
            y.mapv_inplace(move |x2| half * (x2 + F::one()));
            return y;
        } else {
            x.mapv(move |a| ((a * half).tanh() * half) + half)
        }
    }
}

#[inline]
pub fn softmax_impl<T: Float>(x: &NdArrayView<T>, axis: isize) -> NdArray<T> {
    let axis = if axis < 0 {
        (x.ndim() as isize + axis) as usize
    } else {
        axis as usize
    };

    let mut a = x.shape().to_vec();
    a[axis] = 1;
    let reduced_shape = a.as_slice();
    let max_fn = T::max;
    // unwrap is safe
    let max = &x
        .fold_axis(ndarray::Axis(axis), T::min_value(), move |&a, &b| {
            max_fn(a, b)
        })
        .into_shape(ndarray::IxDyn(reduced_shape))
        .unwrap();
    // subtract `max` to prevent overflow
    let mut tmp = x - max;
    #[cfg(feature = "mkl")]
    {
        crate::ops::math_ops::fast_inplace_exp_impl(&mut tmp);
    }
    #[cfg(not(feature = "mkl"))]
    {
        tmp.mapv_inplace(move |a| a.exp());
    }
    // unwrap is safe
    let sum = tmp
        .sum_axis(ndarray::Axis(axis))
        .into_shape(ndarray::IxDyn(reduced_shape))
        .unwrap();
    tmp /= &sum;
    tmp
}

impl<T: Float> op::Op<T> for Softmax {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = softmax_impl(&ctx.input(0), self.axis);
        ctx.append_output(ret)
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let s = ctx.graph();
        let y = ctx.output();
        let gy = ctx.output_grad();
        let sum = s.reduce_sum(y * gy, &[self.axis], true);
        ctx.append_input_grad(Some((gy - sum) * y))
    }
}

impl<T: Float> op::Op<T> for Softplus {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(move |a| (a.exp() + T::one()).ln());
        ctx.append_output(ret)
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gy = ctx.output_grad();
        let s = ctx.graph();
        let a = s.exp(ctx.input(0));
        let b = a + s.scalar(T::one());
        let gx = gy * (a / b);
        ctx.append_input_grad(Some(gx))
    }
}

impl<T: Float> op::Op<T> for Sigmoid {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret;
        #[cfg(feature = "mkl")]
        {
            ret = fast_sigmoid_impl(&ctx.input(0));
        }
        #[cfg(not(feature = "mkl"))]
        {
            let half = T::from(0.5).unwrap();
            ret = ctx
                .input(0)
                .mapv(move |a| ((a * half).tanh() * half) + half);
        }
        ctx.append_output(ret)
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gy = ctx.output_grad();
        let y = ctx.output();
        ctx.append_input_grad(Some(gy * (y - ctx.graph().square(y))));
    }
}

impl<T: Float> op::Op<T> for ReLU {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.max(T::zero()));
        ctx.append_output(ret);
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let s = ctx.graph();
        let gy = ctx.output_grad();
        let bin = s.greater(ctx.input(0), s.scalar(T::zero()));
        ctx.append_input_grad(Some(s.mul(bin, gy)))
    }
}

impl<T: Float> op::Op<T> for Identity {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        // do nothing
        let ret = ctx.input(0);
        ctx.append_output_view(ret)
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gy = ctx.output_grad();
        // use gy's array with rc increment.
        ctx.append_input_grad(Some(gy))
    }
}

impl<T: Float> op::Op<T> for ELU<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).mapv(move |a| {
            if a > T::zero() {
                a
            } else {
                self.alpha * (a.exp() - T::one())
            }
        });
        ctx.append_output(ret)
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gy = &ctx.output_grad();
        let gx = Tensor::builder()
            .set_ro_inputs(&[&ctx.input(0), gy])
            .set_shape(&ctx.graph().shape(gy))
            .build(ctx.graph(), ELUGrad { alpha: self.alpha });
        ctx.append_input_grad(Some(gx))
    }
}

impl<T: Float> op::Op<T> for ELUGrad<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let x = &ctx.input(0);
        let a = x.mapv(move |a| {
            if a > T::zero() {
                T::one()
            } else {
                self.alpha * (a.exp() - T::one()) + self.alpha
            }
        });
        let ret = a * &ctx.input(1);
        ctx.append_output(ret)
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None);
        ctx.append_input_grad(None);
    }
}
