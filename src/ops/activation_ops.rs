use ndarray;
use ndarray_ext::NdArray;
use op;
use ops;
use tensor::Tensor;
use Float;

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

#[inline]
pub fn softmax_forward<T: Float>(x: &NdArray<T>, axis: isize) -> NdArray<T> {
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
    let ref max = x
        .fold_axis(ndarray::Axis(axis), T::min_value(), move |&a, &b| {
            max_fn(a, b)
        })
        .into_shape(ndarray::IxDyn(reduced_shape))
        .unwrap();
    // subtract `max` to prevent overflow
    let mut tmp = x - max;
    tmp.mapv_inplace(|a| a.exp());
    // unwrap is safe
    let sum = tmp
        .sum_axis(ndarray::Axis(axis))
        .into_shape(ndarray::IxDyn(reduced_shape))
        .unwrap();
    tmp /= &sum;
    tmp
}

impl<T: Float> op::Op<T> for Softmax {
    fn name(&self) -> &str {
        "Softmax"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        vec![Ok(softmax_forward(ctx.grab_inputs()[0], self.axis))]
    }

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], output: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let sum = ops::reduce_sum(&(output * gy), &[self.axis], true);
        vec![Some((gy - sum) * output)]
    }
}

impl<T: Float> op::Op<T> for Softplus {
    fn name(&self) -> &str {
        "Softplus"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        use std::f64;
        let e = T::from(f64::consts::E).unwrap();
        vec![Ok(xs[0].mapv(move |a| (a.exp() + T::one()).log(e)))]
    }

    fn grad(&self, gy: &Tensor<T>, xs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let a = &ops::exp(xs[0]);
        let b = a + ops::scalar(T::one());
        let gx = gy * (a / b);
        vec![Some(gx)]
    }
}

impl<T: Float> op::Op<T> for Sigmoid {
    fn name(&self) -> &str {
        "Sigmoid"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        let half = T::from(0.5).unwrap();
        vec![Ok(x.mapv(move |a| ((a * half).tanh() * half) + half))]
    }

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], y: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(gy * (y - ops::square(y)))]
    }
}

impl<T: Float> op::Op<T> for ReLU {
    fn name(&self) -> &str {
        "ReLU"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.max(T::zero())))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let bin = ops::greater(inputs[0], &ops::scalar(T::zero()));
        // inplace is ok because the second derivative of relu is 0.
        // (`mul_inplace` returns `None` as an input gradient.)
        vec![Some(ops::mul_inplace(bin, gy))]
    }
}

impl<T: Float> op::Op<T> for Identity {
    fn name(&self) -> &str {
        "Identity"
    }

    fn compute(&self, _: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        // do nothing
        vec![Err(::op::ComputeException::Delegate { to: 0 })]
    }

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        // use gy's array with rc increment.
        vec![Some(gy.clone())]
    }
}

impl<T: Float> op::Op<T> for ELU<T> {
    fn name(&self) -> &str {
        "ELU"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        let ret = x.mapv(move |a| {
            if a > T::zero() {
                a
            } else {
                self.alpha * (a.exp() - T::one())
            }
        });
        vec![Ok(ret)]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let gx = Tensor::builder()
            .set_inputs(vec![inputs[0], gy])
            .set_shape(gy.shape())
            .build(ELUGrad { alpha: self.alpha });
        vec![Some(gx)]
    }
}

impl<T: Float> op::Op<T> for ELUGrad<T> {
    fn name(&self) -> &str {
        "ELUGrad"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let gy = xs[1];
        let a = x.mapv(move |a| {
            if a > T::zero() {
                T::one()
            } else {
                self.alpha * (a.exp() - T::one()) + self.alpha
            }
        });
        vec![Ok(a * gy)]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None, None]
    }
}
