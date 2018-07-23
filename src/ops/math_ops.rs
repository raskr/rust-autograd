use ndarray;
use ndarray::Zip;
use ndarray_ext::NdArray;
use op;
use ops;
use std::f32;
use std::ops::Neg;
use tensor::Tensor;

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
pub struct Log {
    pub a: f32,
}
pub struct Pow {
    pub a: f32,
}
pub struct LogSumExp {
    pub axis: isize,
    pub keep_dims: bool,
}
pub struct Transpose {
    pub zip: bool,
}

#[inline(always)]
fn equal(a: f32, b: f32) -> f32 {
    ((a == b) as i32) as f32
}
#[inline(always)]
fn not_equal(a: f32, b: f32) -> f32 {
    ((a != b) as i32) as f32
}
#[inline(always)]
fn greater(a: f32, b: f32) -> f32 {
    ((a > b) as i32) as f32
}
#[inline(always)]
fn lesser(a: f32, b: f32) -> f32 {
    ((a < b) as i32) as f32
}
#[inline(always)]
fn greater_equal(a: f32, b: f32) -> f32 {
    ((a >= b) as i32) as f32
}
#[inline(always)]
fn lesser_equal(a: f32, b: f32) -> f32 {
    ((a <= b) as i32) as f32
}
#[inline(always)]
fn maximum(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}
#[inline(always)]
fn minimum(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}

macro_rules! impl_cmp_op {
    ($struct_name:ident, $assign:expr, $grad_fn:expr) => {
        pub struct $struct_name;

        impl op::Op for $struct_name {
            fn name(&self) -> &str {
                stringify!($struct_name)
            }

            fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
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

            fn grad(&self, gy: &Tensor, xs: &[&Tensor], y: &Tensor) -> Vec<Option<Tensor>> {
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

fn none_grad(_: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
    vec![None]
}

fn min_max_grad(gy: &Tensor, xs: &[&Tensor], y: &Tensor) -> Vec<Option<Tensor>> {
    let a = xs[0];
    let b = xs[1];
    let selected_a = ops::equal(a, y);
    let selected_b = ops::equal(b, y);
    vec![
        Some(ops::mul_inplace(selected_a, gy)),
        Some(ops::mul_inplace(selected_b, gy)),
    ]
}

impl op::Op for Abs {
    fn name(&self) -> &str {
        "Abs"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let xs = ctx.grab_inputs();
        vec![Ok(xs[0].map(|x| x.abs()))]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(gy * ops::sign(inputs[0]))]
    }
}

impl op::Op for NegOp {
    fn name(&self) -> &str {
        "Neg"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let xs = ctx.grab_inputs();
        vec![Ok(xs[0].map(|x| x.neg()))]
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(ops::neg(gy))]
    }
}

impl op::Op for Square {
    fn name(&self) -> &str {
        "Square"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let xs = ctx.grab_inputs();
        vec![Ok(xs[0].map(|x| x * x))]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(2 * inputs[0] * gy)]
    }
}

impl op::Op for Reciprocal {
    fn name(&self) -> &str {
        "Reciprocal"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let xs = ctx.grab_inputs();
        vec![Ok(xs[0].map(|x| x.recip()))]
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(ops::neg(&ops::square(output)) * gy)]
    }
}

impl op::Op for Sign {
    fn name(&self) -> &str {
        "Sign"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let xs = ctx.grab_inputs();
        vec![Ok(xs[0].mapv(|x| {
            if x == 0. {
                0.
            } else {
                x.signum()
            }
        }))]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        vec![None]
    }
}

impl op::Op for Floor {
    fn name(&self) -> &str {
        "Floor"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let xs = ctx.grab_inputs();
        vec![Ok(xs[0].map(|x| x.floor()))]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        vec![None]
    }
}

impl op::Op for Ceil {
    fn name(&self) -> &str {
        "Ceil"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let xs = ctx.grab_inputs();
        vec![Ok(xs[0].map(|x| x.ceil()))]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        vec![None]
    }
}

impl op::Op for Transpose {
    fn name(&self) -> &str {
        "Transpose"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let xs = ctx.grab_inputs();
        let x = xs[0].view();
        let perm: &NdArray = &xs[1];
        assert!(perm.len() >= 2);

        let ret = if transpose_reversed(perm) {
            Ok(xs[0].clone().reversed_axes())
        } else {
            // preprocess
            let src_dst = if self.zip {
                perm.iter()
                    .map(|&a| a as usize)
                    .zip(0..perm.len())
                    .collect::<Vec<_>>()
            } else {
                let mut a = perm.iter()
                    .map(|&a| a as usize)
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

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        let gx = Tensor::builder()
            .set_inputs(vec![gy, inputs[1]])
            .set_shape(inputs[0].shape())
            .build(Transpose { zip: !self.zip });
        vec![Some(gx), None]
    }
}

fn do_transpose(mut x: ::ndarray_ext::NdArrayView, mut src_dst: Vec<(usize, usize)>) -> NdArray {
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
fn transpose_reversed(perm: &NdArray) -> bool {
    use std::f32;
    let mut last = f32::MAX;
    for a in perm.iter() {
        if *a > last {
            return false;
        }
        last = *a
    }
    true
}

pub fn logsumexp_forward(x: &NdArray, axis: isize, keep_dims: bool) -> NdArray {
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

    let max_fn = f32::max;
    let ref max = x.fold_axis(ndarray::Axis(axis), f32::MIN, move |&a, &b| max_fn(a, b))
        .into_shape(ndarray::IxDyn(reduced_shape))
        .unwrap();

    // subtract `max` to prevent overflow of exp
    let mut tmp = x - max;

    let exp = {
        tmp.mapv_inplace(|a| a.exp());
        tmp
    };

    // unwrap is safe
    let mut sum = exp.sum_axis(ndarray::Axis(axis))
        .into_shape(ndarray::IxDyn(reduced_shape))
        .unwrap();

    let e = f32::consts::E;
    sum.mapv_inplace(move |a| a.log(e));
    sum += max;
    sum
}

impl op::Op for LogSumExp {
    fn name(&self) -> &str {
        "LogSumExp"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let x = ctx.grab_inputs()[0];
        vec![Ok(logsumexp_forward(x, self.axis, self.keep_dims))]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        // let ref sum = ops::exp(output);
        // let ref exp = ops::exp(inputs[0]);
        // let gx = gy * exp / sum;
        let gx = ops::softmax(inputs[0], self.axis) * gy;
        vec![Some(gx)]
    }
}

impl op::Op for Pow {
    fn name(&self) -> &str {
        "Pow"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let x0 = ctx.grab_inputs()[0];
        let a = self.a;
        vec![Ok(x0.map(move |x| x.powf(a)))]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        let x = inputs[0];
        let gx = gy * self.a * ops::pow(x, self.a - 1.);
        vec![Some(gx)]
    }
}

impl op::Op for Sqrt {
    fn name(&self) -> &str {
        "Sqrt"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let x0 = ctx.grab_inputs()[0];
        vec![Ok(x0.map(|a| a.sqrt()))]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        let x = inputs[0];
        let ret = 0.5 * ops::pow(x, -0.5);
        vec![Some(gy * ret)]
    }
}

impl op::Op for Log {
    fn name(&self) -> &str {
        "Log"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(move |a| a.log(self.a)))]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(gy / inputs[0])]
    }
}

impl op::Op for Exp {
    fn name(&self) -> &str {
        "Exp"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.exp()))]
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(output * gy)]
    }
}

impl op::Op for Atanh {
    fn name(&self) -> &str {
        "Atanh"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        let x = inputs[0];
        let y = ops::reciprocal(&(1 - ops::square(x)));
        vec![Some(y * gy)]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.atanh()))]
    }
}

impl op::Op for Acosh {
    fn name(&self) -> &str {
        "Acosh"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        let x = inputs[0];
        let y = -1 / ops::sqrt(&(ops::square(x) - 1));
        vec![Some(y * gy)]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.acosh()))]
    }
}

impl op::Op for Asinh {
    fn name(&self) -> &str {
        "Asinh"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        let x = inputs[0];
        let y = 1 / ops::sqrt(&(x * x + 1));
        vec![Some(y * gy)]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.asinh()))]
    }
}

impl op::Op for Tanh {
    fn name(&self) -> &str {
        "Tanh"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.tanh()))]
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], y: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(gy * (1 - ops::square(y)))]
    }
}

impl op::Op for Cosh {
    fn name(&self) -> &str {
        "Cosh"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(ops::sinh(inputs[0]) * gy)]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.cosh()))]
    }
}

impl op::Op for Sinh {
    fn name(&self) -> &str {
        "Sinh"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(ops::cosh(inputs[0]) * gy)]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.sinh()))]
    }
}

impl op::Op for Atan {
    fn name(&self) -> &str {
        "Atan"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        let x = inputs[0];
        let y = ops::reciprocal(&(1 + ops::square(x)));
        vec![Some(y * gy)]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.atan()))]
    }
}

impl op::Op for Acos {
    fn name(&self) -> &str {
        "Acos"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        let x = inputs[0];
        let y = -1 / ops::sqrt(&(1 - ops::square(x)));
        vec![Some(y * gy)]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.acos()))]
    }
}

impl op::Op for Asin {
    fn name(&self) -> &str {
        "Asin"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        let x = inputs[0];
        let y = 1 / ops::sqrt(&(1 - x * x));
        vec![Some(y * gy)]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.asin()))]
    }
}

impl op::Op for Sin {
    fn name(&self) -> &str {
        "Sin"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(ops::cos(inputs[0]) * gy)]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.sin()))]
    }
}

impl op::Op for Cos {
    fn name(&self) -> &str {
        "Cos"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(ops::neg(&(ops::sin(inputs[0]) * gy)))]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.cos()))]
    }
}

impl op::Op for Tan {
    fn name(&self) -> &str {
        "Tan"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        let ref cos = ops::cos(inputs[0]);
        vec![Some(gy / (ops::square(cos)))]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let x = ctx.grab_inputs()[0];
        vec![Ok(x.map(|a| a.tan()))]
    }
}
