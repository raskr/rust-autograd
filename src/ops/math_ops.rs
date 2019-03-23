use ndarray;
use ndarray::Zip;
use ndarray_ext::NdArray;
use op;
use ops;
use tensor::Tensor;
use Float;

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
    pub zip: bool,
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
    ($struct_name:ident, $assign:expr, $grad_fn:expr) => {
        pub struct $struct_name;

        impl<T: Float> op::Op<T> for $struct_name {
            fn name(&self) -> &str {
                stringify!($struct_name)
            }

            fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
                let xs = ctx.grab_inputs();
                let x0 = xs[0];
                let x1 = xs[1];
                let shape0 = x0.shape();
                let shape1 = x1.shape();

                let x0_is_scalar = ::ndarray_ext::is_scalar_shape(shape0);
                let x1_is_scalar = ::ndarray_ext::is_scalar_shape(shape1);

                let ret = if x0_is_scalar && x1_is_scalar {
                    let x1_elem = x1[ndarray::IxDyn(&[])];
                    Ok(x0.map(move |a| $assign(a.clone(), x1_elem)))
                } else if x0_is_scalar && !x1_is_scalar {
                    let x0_elem = x0[ndarray::IxDyn(&[])];
                    Ok(x1.map(move |a| $assign(x0_elem, a.clone())))
                } else if !x0_is_scalar && x1_is_scalar {
                    let x1_elem = x1[ndarray::IxDyn(&[])];
                    Ok(x0.map(move |a| $assign(a.clone(), x1_elem)))
                } else {
                    // case that scalar is not involved
                    // Check the input ranks.
                    // op couldn't we catch here cause ndarray's panics.

                    // rank check
                    if shape0.len() != shape1.len() {
                        let name0 = ctx.grab_input_node(0).op.name();
                        let name1 = ctx.grab_input_node(1).op.name();
                        panic!(
                            "Tensor ranks mismatch: {}({}) vs {}({})",
                            shape0.len(),
                            name0,
                            shape1.len(),
                            name1
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
                        Ok(result)
                    } else if size0 > size1 {
                        let name0 = &ctx.grab_input_node(0).op.name();
                        let name1 = &ctx.grab_input_node(1).op.name();
                        panic!(
                            "Tensor ranks mismatch: {}({}) vs {}({})",
                            shape0.len(),
                            name0,
                            shape1.len(),
                            name1
                        );
                    } else {
                        // same
                        let mut result = NdArray::zeros(shape0);
                        Zip::from(&mut result)
                            .and(x0)
                            .and(x1)
                            .apply(|r, a, b| *r = $assign(a.clone(), b.clone()));
                        Ok(result)
                    }
                };

                vec![ret]
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

impl_cmp_op!(Equal, equal, none_grad);
impl_cmp_op!(NotEqual, not_equal, none_grad);
impl_cmp_op!(Greater, greater, none_grad);
impl_cmp_op!(Lesser, lesser, none_grad);
impl_cmp_op!(GreaterEqual, greater_equal, none_grad);
impl_cmp_op!(LesserEqual, lesser_equal, none_grad);
impl_cmp_op!(Maximum, maximum, min_max_grad);
impl_cmp_op!(Minimum, minimum, min_max_grad);

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
        Some(ops::mul_inplace(selected_a, gy)),
        Some(ops::mul_inplace(selected_b, gy)),
    ]
}

impl<T: Float> op::Op<T> for Abs {
    fn name(&self) -> &str {
        "Abs"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        vec![Ok(xs[0].map(|x| x.abs()))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(gy * ops::sign(inputs[0]))]
    }
}

impl<T: Float> op::Op<T> for NegOp {
    fn name(&self) -> &str {
        "Neg"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        vec![Ok(xs[0].map(|x| x.neg()))]
    }

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(ops::neg(gy))]
    }
}

impl<T: Float> op::Op<T> for Square {
    fn name(&self) -> &str {
        "Square"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        vec![Ok(xs[0].map(|&x| x * x))]
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

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        vec![Ok(xs[0].map(|x| x.recip()))]
    }

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], output: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(ops::neg(&ops::square(output)) * gy)]
    }
}

impl<T: Float> op::Op<T> for Sign {
    fn name(&self) -> &str {
        "Sign"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        vec![Ok(xs[0].mapv(|x| {
            if x == T::zero() {
                T::zero()
            } else {
                x.signum()
            }
        }))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<T: Float> op::Op<T> for Floor {
    fn name(&self) -> &str {
        "Floor"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        vec![Ok(xs[0].map(|x| x.floor()))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<T: Float> op::Op<T> for Ceil {
    fn name(&self) -> &str {
        "Ceil"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        vec![Ok(xs[0].map(|x| x.ceil()))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<T: Float> op::Op<T> for Transpose {
    fn name(&self) -> &str {
        "Transpose"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x = xs[0].view();
        let perm: &NdArray<T> = &xs[1];
        assert!(perm.len() >= 2);

        let ret = if transpose_reversed(perm) {
            Ok(xs[0].clone().reversed_axes())
        } else {
            // preprocess
            let src_dst = if self.zip {
                perm.iter()
                    .map(|&a| a.to_usize().unwrap())
                    .zip(0..perm.len())
                    .collect::<Vec<_>>()
            } else {
                let mut a = perm
                    .iter()
                    .map(|&a| a.to_usize().unwrap())
                    .enumerate()
                    .collect::<Vec<_>>();
                a.sort_by_key(|sd| sd.1);
                a
            };

            // permutes dimensions
            Ok(do_transpose(x, src_dst))
        };
        vec![ret]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let gx = Tensor::builder()
            .set_inputs(vec![gy, inputs[1]])
            .set_shape(inputs[0].shape())
            .build(Transpose { zip: !self.zip });
        vec![Some(gx), None]
    }
}

fn do_transpose<T: Float>(
    mut x: ::ndarray_ext::NdArrayView<T>,
    mut src_dst: Vec<(usize, usize)>,
) -> NdArray<T> {
    for i in 0..src_dst.len() {
        let (src, dst) = {
            let sd = src_dst[i];
            (sd.0, sd.1)
        };

        if src <= dst {
            continue;
        }

        for j in (dst..src).rev() {
            // "bigger to smaller" iteration is important
            x.swap_axes(j, j + 1); // Swaps two axes
                                   // Increments "src"es I passed by.
            for sd in src_dst.iter_mut() {
                if sd.0 == j {
                    sd.0 += 1;
                    break;
                }
            }
        }

        src_dst[i].0 = dst;
    }
    if x.is_standard_layout() {
        x.to_owned()
    } else {
        NdArray::from_shape_fn(x.shape(), |i| x[i])
    }
}

// Helper for transpose. Returns true if axes are just reversed
fn transpose_reversed<T: Float>(perm: &NdArray<T>) -> bool {
    let mut last = T::max_value();
    for a in perm.iter() {
        if *a > last {
            return false;
        }
        last = *a
    }
    true
}

pub fn logsumexp_forward<T: Float>(x: &NdArray<T>, axis: isize, keep_dims: bool) -> NdArray<T> {
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

    // subtract `max` to prevent overflow of exp
    let mut tmp = x - max;

    let exp = {
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

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        vec![Ok(logsumexp_forward(x, self.axis, self.keep_dims))]
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

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x0 = ctx.grab_inputs()[0];
        let a = self.a;
        vec![Ok(x0.map(move |x| x.powf(a)))]
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

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x0 = ctx.grab_inputs()[0];
        vec![Ok(x0.map(|a| a.sqrt()))]
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

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(move |a| a.log(self.a)))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(gy / inputs[0])]
    }
}

impl<T: Float> op::Op<T> for Exp {
    fn name(&self) -> &str {
        "Exp"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.exp()))]
    }

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], output: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(output * gy)]
    }
}

impl<T: Float> op::Op<T> for Atanh {
    fn name(&self) -> &str {
        "Atanh"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.atanh()))]
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

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.acosh()))]
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

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.asinh()))]
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

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.tanh()))]
    }

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], y: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(gy * (ops::scalar(T::one()) - ops::square(y)))]
    }
}

impl<T: Float> op::Op<T> for Cosh {
    fn name(&self) -> &str {
        "Cosh"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.cosh()))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(ops::sinh(inputs[0]) * gy)]
    }
}

impl<T: Float> op::Op<T> for Sinh {
    fn name(&self) -> &str {
        "Sinh"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.sinh()))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(ops::cosh(inputs[0]) * gy)]
    }
}

impl<T: Float> op::Op<T> for Atan {
    fn name(&self) -> &str {
        "Atan"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.atan()))]
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

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.acos()))]
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

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.asin()))]
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

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.sin()))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(ops::cos(inputs[0]) * gy)]
    }
}

impl<T: Float> op::Op<T> for Cos {
    fn name(&self) -> &str {
        "Cos"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.cos()))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(ops::neg(&(ops::sin(inputs[0]) * gy)))]
    }
}

impl<T: Float> op::Op<T> for Tan {
    fn name(&self) -> &str {
        "Tan"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.tan()))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let ref cos = ops::cos(inputs[0]);
        vec![Some(gy / (ops::square(cos)))]
    }
}
