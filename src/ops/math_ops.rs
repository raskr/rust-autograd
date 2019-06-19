use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::ops;
use crate::tensor::Tensor;
use crate::Float;
use ndarray;
use ndarray::Zip;

pub struct Sin;
pub struct Cos;
pub struct Tan;
pub struct Asin;
pub struct Acos;
pub struct Atan;
pub struct Sinh;
pub struct Cosh;
pub struct Tanh;
pub struct Asinh;
pub struct Acosh;
pub struct Atanh;
pub struct Exp;
pub struct Sqrt;
pub struct NegOp;
pub struct Floor;
pub struct Ceil;
pub struct Sign;
pub struct Reciprocal;
pub struct Square;
pub struct Abs;
pub struct Log<T: Float> {
    pub a: T,
}
pub struct Pow<T: Float> {
    pub a: T,
}
pub struct LogSumExp {
    pub axis: isize,
    pub keep_dims: bool,
}
pub struct Transpose {
    pub invert_axes: bool,
}

#[inline(always)]
fn equal<T: Float>(a: T, b: T) -> T {
    T::from((a == b) as i32).unwrap()
}
#[inline(always)]
fn not_equal<T: Float>(a: T, b: T) -> T {
    T::from((a != b) as i32).unwrap()
}
#[inline(always)]
fn greater<T: Float>(a: T, b: T) -> T {
    T::from((a > b) as i32).unwrap()
}
#[inline(always)]
fn lesser<T: Float>(a: T, b: T) -> T {
    T::from((a < b) as i32).unwrap()
}
#[inline(always)]
fn greater_equal<T: Float>(a: T, b: T) -> T {
    T::from((a >= b) as i32).unwrap()
}
#[inline(always)]
fn lesser_equal<T: Float>(a: T, b: T) -> T {
    T::from((a <= b) as i32).unwrap()
}
#[inline(always)]
fn maximum<T: Float>(a: T, b: T) -> T {
    a.max(b)
}
#[inline(always)]
fn minimum<T: Float>(a: T, b: T) -> T {
    a.min(b)
}

macro_rules! impl_cmp_op {
    ($struct_name:ident, $name:expr, $assign:expr, $grad_fn:expr) => {
        pub struct $struct_name;

        impl<T: Float> op::Op<T> for $struct_name {
            fn name(&self) -> &str {
                stringify!($struct_name)
            }

            fn compute<'v>(
                &self,
                ctx: crate::runtime::OpComputeContext<'v, T>,
            ) -> op::ComputeResults<'v, T> {
                let xs = ctx.grab_inputs();
                let x0 = &xs[0];
                let x1 = &xs[1];
                let shape0 = x0.shape();
                let shape1 = x1.shape();

                let x0_is_scalar = crate::ndarray_ext::is_scalar_shape(shape0);
                let x1_is_scalar = crate::ndarray_ext::is_scalar_shape(shape1);

                let ret = if x0_is_scalar && x1_is_scalar {
                    let x1_elem = x1[ndarray::IxDyn(&[])];
                    x0.map(move |a| $assign(a.clone(), x1_elem))
                } else if x0_is_scalar && !x1_is_scalar {
                    let x0_elem = x0[ndarray::IxDyn(&[])];
                    x1.map(move |a| $assign(x0_elem, a.clone()))
                } else if !x0_is_scalar && x1_is_scalar {
                    let x1_elem = x1[ndarray::IxDyn(&[])];
                    x0.map(move |a| $assign(a.clone(), x1_elem))
                } else {
                    // case that scalar is not involved
                    // Check the input ranks.
                    // op couldn't we catch here cause ndarray's panics.

                    // rank check
                    if shape0.len() != shape1.len() {
                        panic!(
                            "Tensor ranks mismatch: {}({}'s lhs input) vs {}({}'s rhs input)",
                            shape0.len(),
                            $name,
                            shape1.len(),
                            $name,
                        )
                    }

                    let size0: usize = shape0.iter().product();
                    let size1: usize = shape1.iter().product();

                    // Whether broadcast of x0 and x1 is needed or not is depends on
                    // their shapes.
                    // FIXME: Is this cond branch ok?
                    if size0 < size1 {
                        let mut result = NdArray::zeros(shape1);
                        Zip::from(&mut result)
                            .and_broadcast(x0)
                            .and(x1)
                            .apply(|r, a, b| *r = $assign(a.clone(), b.clone()));
                        result
                    } else if size0 > size1 {
                        panic!(
                            "Tensor ranks mismatch: {}({}'s lhs input) vs {}({}'s rhs input)",
                            shape0.len(),
                            $name,
                            shape1.len(),
                            $name
                        );
                    } else {
                        // same
                        let mut result = NdArray::zeros(shape0);
                        Zip::from(&mut result)
                            .and(x0)
                            .and(x1)
                            .apply(|r, a, b| *r = $assign(a.clone(), b.clone()));
                        result
                    }
                };

                vec![Ok(crate::ArrRepr::Owned(ret))]
            }

            fn grad(
                &self,
                gy: &Tensor<T>,
                xs: &[&Tensor<T>],
                y: &Tensor<T>,
            ) -> Vec<Option<Tensor<T>>> {
                $grad_fn(gy, xs, y)
            }
        }
    };
}

impl_cmp_op!(Equal, "Equal", equal, none_grad);
impl_cmp_op!(NotEqual, "NotEqual", not_equal, none_grad);
impl_cmp_op!(Greater, "Greater", greater, none_grad);
impl_cmp_op!(Lesser, "Lesser", lesser, none_grad);
impl_cmp_op!(GreaterEqual, "GreaterEqual", greater_equal, none_grad);
impl_cmp_op!(LesserEqual, "LesserEqual", lesser_equal, none_grad);
impl_cmp_op!(Maximum, "Maximum", maximum, min_max_grad);
impl_cmp_op!(Minimum, "Minimum", minimum, min_max_grad);

fn none_grad<T: Float>(_: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
    vec![None]
}

fn min_max_grad<T: Float>(
    gy: &Tensor<T>,
    xs: &[&Tensor<T>],
    y: &Tensor<T>,
) -> Vec<Option<Tensor<T>>> {
    let a = xs[0];
    let b = xs[1];
    let selected_a = ops::equal(a, y);
    let selected_b = ops::equal(b, y);
    vec![
        Some(ops::mul(selected_a, gy)),
        Some(ops::mul(selected_b, gy)),
    ]
}

impl<T: Float> op::Op<T> for Abs {
    fn name(&self) -> &str {
        "Abs"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        vec![Ok(crate::ArrRepr::Owned(xs[0].map(|x| x.abs())))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(gy * ops::sign(inputs[0]))]
    }
}

impl<T: Float> op::Op<T> for NegOp {
    fn name(&self) -> &str {
        "Neg"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        vec![Ok(crate::ArrRepr::Owned(xs[0].map(|x| x.neg())))]
    }

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(ops::neg(gy))]
    }
}

impl<T: Float> op::Op<T> for Square {
    fn name(&self) -> &str {
        "Square"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        vec![Ok(crate::ArrRepr::Owned(xs[0].map(|&x| x * x)))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let two = ops::scalar(T::one() + T::one());
        vec![Some(two * inputs[0] * gy)]
    }
}

impl<T: Float> op::Op<T> for Reciprocal {
    fn name(&self) -> &str {
        "Reciprocal"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        vec![Ok(crate::ArrRepr::Owned(xs[0].map(|x| x.recip())))]
    }

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], output: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(ops::neg(&ops::square(output)) * gy)]
    }
}

impl<T: Float> op::Op<T> for Sign {
    fn name(&self) -> &str {
        "Sign"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        vec![Ok(crate::ArrRepr::Owned(xs[0].mapv(|x| {
            if x == T::zero() {
                T::zero()
            } else {
                x.signum()
            }
        })))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<T: Float> op::Op<T> for Floor {
    fn name(&self) -> &str {
        "Floor"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        vec![Ok(crate::ArrRepr::Owned(xs[0].map(|x| x.floor())))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<T: Float> op::Op<T> for Ceil {
    fn name(&self) -> &str {
        "Ceil"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        vec![Ok(crate::ArrRepr::Owned(xs[0].map(|x| x.ceil())))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<T: Float> op::Op<T> for Transpose {
    fn name(&self) -> &str {
        "Transpose"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        let perm = &xs[1];
        let perm_len = perm.len();
        assert!(perm_len >= 2);

        let ret = {
            let mut dims = crate::uninitialized_vec::<usize>(perm_len);
            for (i, d) in perm.iter().enumerate() {
                if self.invert_axes {
                    dims[d.to_usize().unwrap()] = i;
                } else {
                    dims[i] = d.to_usize().unwrap();
                }
            }
            xs[0].clone().permuted_axes(dims.as_slice())
        };

        vec![Ok(crate::ArrRepr::View(ret))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let gx = Tensor::builder()
            .set_inputs(vec![gy, inputs[1]])
            .set_shape(inputs[0].shape())
            .build(Transpose {
                invert_axes: !self.invert_axes,
            });
        vec![Some(gx), None]
    }
}

pub fn logsumexp_forward<T: Float>(x: &NdArrayView<T>, axis: isize, keep_dims: bool) -> NdArray<T> {
    let axis = if axis < 0 {
        (x.ndim() as isize + axis) as usize
    } else {
        axis as usize
    };

    let mut a = x.shape().to_vec();
    if keep_dims {
        a[axis] = 1;
    } else {
        a.remove(axis);
    }
    let reduced_shape = a.as_slice();

    let max_fn = T::max;
    let min_val = T::min_value();
    let ref max = x
        .fold_axis(ndarray::Axis(axis), min_val, move |&a, &b| max_fn(a, b))
        .into_shape(ndarray::IxDyn(reduced_shape))
        .unwrap();

    let exp = {
        // subtract `max` to prevent overflow of exp
        let mut tmp = x - max;
        tmp.mapv_inplace(|a| a.exp());
        tmp
    };

    // unwrap is safe
    let mut sum = exp
        .sum_axis(ndarray::Axis(axis))
        .into_shape(ndarray::IxDyn(reduced_shape))
        .unwrap();

    use std::f64;
    let e = T::from(f64::consts::E).unwrap();
    sum.mapv_inplace(move |a| a.log(e));
    sum += max;
    sum
}

impl<T: Float> op::Op<T> for LogSumExp {
    fn name(&self) -> &str {
        "LogSumExp"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let x = &ctx.grab_inputs()[0];
        vec![Ok(crate::ArrRepr::Owned(logsumexp_forward(
            x,
            self.axis,
            self.keep_dims,
        )))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        // let ref sum = ops::exp(output);
        // let ref exp = ops::exp(inputs[0]);
        // let gx = gy * exp / sum;
        let gx = ops::softmax(inputs[0], self.axis) * gy;
        vec![Some(gx)]
    }
}

impl<T: Float> op::Op<T> for Pow<T> {
    fn name(&self) -> &str {
        "Pow"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let x0 = &ctx.grab_inputs()[0];
        let a = self.a;
        vec![Ok(crate::ArrRepr::Owned(x0.map(move |x| x.powf(a))))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let x = inputs[0];
        let gx = gy * self.a * ops::pow(x, self.a - T::one());
        vec![Some(gx)]
    }
}

impl<T: Float> op::Op<T> for Sqrt {
    fn name(&self) -> &str {
        "Sqrt"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let x0 = &ctx.grab_inputs()[0];
        vec![Ok(crate::ArrRepr::Owned(x0.map(|a| a.sqrt())))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let x = inputs[0];
        let half = ops::scalar(T::one());
        let ret = half * ops::pow(x, T::one().neg());
        vec![Some(gy * ret)]
    }
}

impl<T: Float> op::Op<T> for Log<T> {
    fn name(&self) -> &str {
        "Log"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let x = &ctx.grab_inputs()[0];
        vec![Ok(crate::ArrRepr::Owned(x.map(move |a| a.log(self.a))))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(gy / inputs[0])]
    }
}

impl<T: Float> op::Op<T> for Exp {
    fn name(&self) -> &str {
        "Exp"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let x = &ctx.grab_inputs()[0];
        vec![Ok(crate::ArrRepr::Owned(x.map(|a| a.exp())))]
    }

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], output: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(output * gy)]
    }
}

impl<T: Float> op::Op<T> for Atanh {
    fn name(&self) -> &str {
        "Atanh"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let x = &ctx.grab_inputs()[0];
        vec![Ok(crate::ArrRepr::Owned(x.map(|a| a.atanh())))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let x = inputs[0];
        let one = ops::scalar(T::one());
        let y = ops::reciprocal(&(one - ops::square(x)));
        vec![Some(y * gy)]
    }
}

impl<T: Float> op::Op<T> for Acosh {
    fn name(&self) -> &str {
        "Acosh"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let x = &ctx.grab_inputs()[0];
        vec![Ok(crate::ArrRepr::Owned(x.map(|a| a.acosh())))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let x = inputs[0];
        let one = &ops::scalar(T::one().neg());
        let y = one / ops::sqrt(&(ops::square(x) + one));
        vec![Some(y * gy)]
    }
}

impl<T: Float> op::Op<T> for Asinh {
    fn name(&self) -> &str {
        "Asinh"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let x = &ctx.grab_inputs()[0];
        vec![Ok(crate::ArrRepr::Owned(x.map(|a| a.asinh())))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let x = inputs[0];
        let one = &ops::scalar(T::one());
        let y = one / ops::sqrt(&(x * x + one));
        vec![Some(y * gy)]
    }
}

impl<T: Float> op::Op<T> for Tanh {
    fn name(&self) -> &str {
        "Tanh"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let x = &ctx.grab_inputs()[0];
        vec![Ok(crate::ArrRepr::Owned(x.map(|a| a.tanh())))]
    }

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], y: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(gy * (ops::scalar(T::one()) - ops::square(y)))]
    }
}

impl<T: Float> op::Op<T> for Cosh {
    fn name(&self) -> &str {
        "Cosh"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let x = &ctx.grab_inputs()[0];
        vec![Ok(crate::ArrRepr::Owned(x.map(|a| a.cosh())))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(ops::sinh(inputs[0]) * gy)]
    }
}

impl<T: Float> op::Op<T> for Sinh {
    fn name(&self) -> &str {
        "Sinh"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let x = &ctx.grab_inputs()[0];
        vec![Ok(crate::ArrRepr::Owned(x.map(|a| a.sinh())))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(ops::cosh(inputs[0]) * gy)]
    }
}

impl<T: Float> op::Op<T> for Atan {
    fn name(&self) -> &str {
        "Atan"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let x = &ctx.grab_inputs()[0];
        vec![Ok(crate::ArrRepr::Owned(x.map(|a| a.atan())))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let x = inputs[0];
        let y = ops::reciprocal(ops::square(x) + T::one());
        vec![Some(y * gy)]
    }
}

impl<T: Float> op::Op<T> for Acos {
    fn name(&self) -> &str {
        "Acos"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let x = &ctx.grab_inputs()[0];
        vec![Ok(crate::ArrRepr::Owned(x.map(|a| a.acos())))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let x = inputs[0];
        let y = ops::scalar(T::one().neg()) / ops::sqrt(&(ops::scalar(T::one()) - ops::square(x)));
        vec![Some(y * gy)]
    }
}

impl<T: Float> op::Op<T> for Asin {
    fn name(&self) -> &str {
        "Asin"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let x = &ctx.grab_inputs()[0];
        vec![Ok(crate::ArrRepr::Owned(x.map(|a| a.asin())))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let x = inputs[0];
        let y = ops::scalar(T::one()) / ops::sqrt(&(ops::scalar(T::one()) - x * x));
        vec![Some(y * gy)]
    }
}

impl<T: Float> op::Op<T> for Sin {
    fn name(&self) -> &str {
        "Sin"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let x = &ctx.grab_inputs()[0];
        vec![Ok(crate::ArrRepr::Owned(x.map(|a| a.sin())))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(ops::cos(inputs[0]) * gy)]
    }
}

impl<T: Float> op::Op<T> for Cos {
    fn name(&self) -> &str {
        "Cos"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let x = &ctx.grab_inputs()[0];
        vec![Ok(crate::ArrRepr::Owned(x.map(|a| a.cos())))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(ops::neg(&(ops::sin(inputs[0]) * gy)))]
    }
}

impl<T: Float> op::Op<T> for Tan {
    fn name(&self) -> &str {
        "Tan"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let x = &ctx.grab_inputs()[0];
        vec![Ok(crate::ArrRepr::Owned(x.map(|a| a.tan())))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let ref cos = ops::cos(inputs[0]);
        vec![Some(gy / (ops::square(cos)))]
    }
}
