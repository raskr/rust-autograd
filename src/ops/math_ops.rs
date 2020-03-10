use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::tensor::Tensor;
use crate::Float;
use crate::Graph;
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
            fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
                let x0 = ctx.input(0);
                let x1 = &ctx.input(1);
                let shape0 = x0.shape();
                let shape1 = x1.shape();

                let x0_is_scalar = crate::ndarray_ext::is_scalar_shape(shape0);
                let x1_is_scalar = crate::ndarray_ext::is_scalar_shape(shape1);

                let ret = if x0_is_scalar && x1_is_scalar {
                    let x1_elem = x1[ndarray::IxDyn(&[])];
                    x0.mapv(move |a| $assign(a, x1_elem))
                } else if x0_is_scalar && !x1_is_scalar {
                    let x0_elem = x0[ndarray::IxDyn(&[])];
                    x1.mapv(move |a| $assign(x0_elem, a))
                } else if !x0_is_scalar && x1_is_scalar {
                    let x1_elem = x1[ndarray::IxDyn(&[])];
                    x0.mapv(move |a| $assign(a, x1_elem))
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

                ctx.append_output(Ok(ret));
            }

            fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
                ctx.set_input_grads($grad_fn(
                    ctx.output_grad(),
                    ctx.input(0),
                    ctx.input(1),
                    ctx.output(),
                    ctx.graph(),
                ));
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

#[inline]
fn none_grad<'a, 'b: 'a, T: Float>(
    _: Tensor<'a, 'b, T>,
    _: Tensor<'a, 'b, T>,
    _: Tensor<'a, 'b, T>,
    _: Tensor<'a, 'b, T>,
    _: &'b Graph<T>,
) -> Vec<Option<Tensor<'a, 'b, T>>> {
    vec![None]
}

#[inline]
fn min_max_grad<'a, 'b: 'a, T: Float>(
    gy: Tensor<'a, 'b, T>,
    x1: Tensor<'a, 'b, T>,
    x2: Tensor<'a, 'b, T>,
    y: Tensor<'a, 'b, T>,
    c: &'b Graph<T>,
) -> Vec<Option<Tensor<'a, 'b, T>>> {
    let selected_a = c.equal(x1, y);
    let selected_b = c.equal(x2, y);
    vec![
        Some(c.mul(selected_a.tensor, gy.tensor)),
        Some(c.mul(selected_b.tensor, gy.tensor)),
    ]
}

impl<T: Float> op::Op<T> for Abs {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|x| x.abs());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![Some(
            ctx.output_grad() * ctx.graph().sign(ctx.input(0)),
        )])
    }
}

impl<T: Float> op::Op<T> for NegOp {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|x| x.neg());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![Some(ctx.graph().neg(ctx.output_grad()))])
    }
}

impl<T: Float> op::Op<T> for Square {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|&x| x * x);
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let two = ctx.graph().scalar(T::one() + T::one());
        ctx.set_input_grads(vec![Some(two * ctx.input(0) * ctx.output_grad())]);
    }
}

impl<T: Float> op::Op<T> for Reciprocal {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|x| x.recip());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![Some(
            ctx.graph().neg(&ctx.graph().square(ctx.output())) * ctx.output_grad(),
        )])
    }
}

impl<T: Float> op::Op<T> for Sign {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).mapv(|x| {
            if x == T::zero() {
                T::zero()
            } else {
                x.signum()
            }
        });
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None]);
    }
}

impl<T: Float> op::Op<T> for Floor {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|x| x.floor());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None]);
    }
}

impl<T: Float> op::Op<T> for Ceil {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|x| x.ceil());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None])
    }
}

impl<T: Float> op::Op<T> for Transpose {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let perm = &ctx.input(1);
        let perm_len = perm.len();
        let x = ctx.input(0);
        assert_eq!(
            x.ndim(),
            perm_len,
            "autograd::transpose: inputs's ndim and axes's length must match"
        );

        let ret = unsafe {
            let mut dims = crate::uninitialized_vec::<usize>(perm_len);
            for (i, d) in perm.iter().enumerate() {
                let d = d.to_usize().unwrap();
                if self.invert_axes {
                    *dims.get_unchecked_mut(d) = i;
                } else {
                    *dims.get_unchecked_mut(i) = d;
                }
            }
            x.permuted_axes(dims.as_slice())
        };

        ctx.append_output_view(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gx = Tensor::builder()
            .set_ro_inputs(&[&ctx.output_grad(), &ctx.input(1)])
            .set_shape(&ctx.graph().shape(&ctx.input(0)))
            .build(
                ctx.graph(),
                Transpose {
                    invert_axes: !self.invert_axes,
                },
            );
        ctx.set_input_grads(vec![Some(gx), None])
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
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = logsumexp_forward(&ctx.input(0), self.axis, self.keep_dims);
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        // let ref sum = c.exp(output);
        // let ref exp = c.exp(ctx.input(0));
        // let gx = gy * exp / sum;
        let gx = ctx.graph().softmax(ctx.input(0), self.axis) * ctx.output_grad();
        ctx.set_input_grads(vec![Some(gx)])
    }
}

impl<T: Float> op::Op<T> for Pow<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let a = self.a;
        let ret = ctx.input(0).map(move |x| x.powf(a));
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let x = ctx.input(0);
        let gx =
            ctx.output_grad() * ctx.graph().scalar(self.a) * ctx.graph().pow(x, self.a - T::one());
        ctx.set_input_grads(vec![Some(gx)])
    }
}

impl<T: Float> op::Op<T> for Sqrt {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.sqrt());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let x = ctx.input(0);
        let half = ctx.graph().scalar(T::one());
        let ret = half * ctx.graph().pow(x, T::one().neg());
        ctx.set_input_grads(vec![Some(ctx.output_grad() * ret)])
    }
}

impl<T: Float> op::Op<T> for Log<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(move |a| a.log(self.a));
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![Some(ctx.output_grad() / ctx.input(0))])
    }
}

impl<T: Float> op::Op<T> for Exp {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.exp());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![Some(ctx.output() * ctx.output_grad())])
    }
}

impl<T: Float> op::Op<T> for Atanh {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.atanh());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let x = ctx.input(0);
        let one = ctx.graph().scalar(T::one());
        let y = ctx.graph().reciprocal(one - ctx.graph().square(x));
        ctx.set_input_grads(vec![Some(y * ctx.output_grad())])
    }
}

impl<T: Float> op::Op<T> for Acosh {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.acosh());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let x = ctx.input(0);
        let one = ctx.graph().scalar(T::one().neg());
        let y = one / ctx.graph().sqrt(ctx.graph().square(x) + one);
        ctx.set_input_grads(vec![Some(y * ctx.output_grad())])
    }
}

impl<T: Float> op::Op<T> for Asinh {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.asinh());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let x = ctx.input(0);
        let one = ctx.graph().scalar(T::one());
        let y = one / ctx.graph().sqrt(x * x + one);
        ctx.set_input_grads(vec![Some(y * ctx.output_grad())])
    }
}

impl<T: Float> op::Op<T> for Tanh {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.tanh());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![Some(
            ctx.output_grad() * (ctx.graph().scalar(T::one()) - ctx.graph().square(ctx.output())),
        )])
    }
}

impl<T: Float> op::Op<T> for Cosh {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.cosh());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![Some(
            ctx.graph().sinh(ctx.input(0)) * ctx.output_grad(),
        )])
    }
}

impl<T: Float> op::Op<T> for Sinh {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.sinh());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![Some(
            ctx.graph().cosh(ctx.input(0)) * ctx.output_grad(),
        )])
    }
}

impl<T: Float> op::Op<T> for Atan {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.atan());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let x = ctx.input(0);
        let y = ctx
            .graph()
            .reciprocal(ctx.graph().square(x) + ctx.graph().scalar(T::one()));
        ctx.set_input_grads(vec![Some(y * ctx.output_grad())])
    }
}

impl<T: Float> op::Op<T> for Acos {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.acos());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let x = ctx.input(0);
        let s = ctx.graph();
        let y =
            ctx.graph().scalar(T::one().neg()) / s.sqrt(ctx.graph().scalar(T::one()) - s.square(x));
        ctx.set_input_grads(vec![Some(y * ctx.output_grad())])
    }
}

impl<T: Float> op::Op<T> for Asin {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.asin());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let x = ctx.input(0);
        let y =
            ctx.graph().scalar(T::one()) / ctx.graph().sqrt(ctx.graph().scalar(T::one()) - x * x);
        ctx.set_input_grads(vec![Some(y * ctx.output_grad())])
    }
}

impl<T: Float> op::Op<T> for Sin {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.sin());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![Some(
            ctx.graph().cos(ctx.input(0)) * ctx.output_grad(),
        )])
    }
}

impl<T: Float> op::Op<T> for Cos {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.cos());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![Some(
            ctx.graph()
                .neg(&(ctx.graph().sin(ctx.input(0)) * ctx.output_grad())),
        )])
    }
}

impl<T: Float> op::Op<T> for Tan {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.tan());
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let cos = ctx.graph().cos(&ctx.input(0));
        ctx.set_input_grads(vec![Some(ctx.output_grad() / ctx.graph().square(cos))])
    }
}
