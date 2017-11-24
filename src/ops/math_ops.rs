extern crate ndarray;

use self::ndarray::Zip;
use ndarray_ext::NdArray;
use ops;
use std::f32;
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
pub struct Floor;
pub struct Ceil;
pub struct Sign;
pub struct Log {
    pub a: f32,
}
pub struct Pow {
    pub a: f32,
}
pub struct LogSumExp {
    pub axis: isize,
}
pub struct Transpose {
    pub zip: bool,
}


macro_rules! impl_cmp_op {
    ($struct_name:ident, $assign:expr) => {

        pub struct $struct_name;

        impl ops::Op for $struct_name {
            fn name(&self) -> &str
            {
                stringify!($struct_name)
            }

            fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
            {
                let x0 = xs[0];
                let x1 = xs[1];
                let shape0 = x0.shape();
                let shape1 = x1.shape();
                let scalar_shape = &[0];

                let x0_is_scalar = shape0 == scalar_shape;
                let x1_is_scalar = !x0_is_scalar;

                if x0_is_scalar {
                    let mut result = NdArray::zeros(shape1);
                    Zip::from(&mut result).and_broadcast(x0).and(x1).apply($assign);
                    Ok(result)
                } else if x1_is_scalar {
                    let mut result = NdArray::zeros(shape0);
                    Zip::from(&mut result).and(x0).and_broadcast(x1).apply($assign);
                    Ok(result)
                } else {
                    let mut result = NdArray::zeros(shape0);
                    Zip::from(&mut result).and(x0).and(x1).apply($assign);
                    Ok(result)
                }
            }

            fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
            {
                vec![None]
            }
        }

    };
}

impl_cmp_op!(Equal, move |r, a, b| *r = ((a == b) as i32) as f32);
impl_cmp_op!(NotEqual, move |r, a, b| *r = ((a != b) as i32) as f32);
impl_cmp_op!(Greater, move |r, a, b| *r = ((a > b) as i32) as f32);
impl_cmp_op!(Lesser, move |r, a, b| *r = ((a < b) as i32) as f32);
impl_cmp_op!(GreaterEqual, move |r, a, b| *r = ((a >= b) as i32) as f32);
impl_cmp_op!(LesserEqual, move |r, a, b| *r = ((a <= b) as i32) as f32);
impl_cmp_op!(Maximum, move |r, a, b| *r = if a > b { *a } else { *b });
impl_cmp_op!(Minimum, move |r, a, b| *r = if a < b { *a } else { *b });


impl ops::Op for Sign {
    fn name(&self) -> &str
    {
        "Sign"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].mapv(|x| if x == 0. { 0. } else { x.signum() }))
    }

    fn grad(&self, _: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

impl ops::Op for Floor {
    fn name(&self) -> &str
    {
        "Floor"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].map(|x| x.floor()))
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

impl ops::Op for Ceil {
    fn name(&self) -> &str
    {
        "Ceil"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].map(|x| x.ceil()))
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

impl ops::Op for Transpose {
    fn name(&self) -> &str
    {
        "Transpose"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x = xs[0].view();
        let perm: &NdArray = &xs[1];
        assert!(perm.len() >= 2);


        if transpose_reversed(perm) {
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
        }
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let op = Transpose { zip: !self.zip };
        vec![
            Some(ops::apply_op(op, &[gy, inputs[1]], Some(inputs[0].shape()))),
            None,
        ]
    }
}

fn do_transpose(mut x: ::ndarray_ext::NdArrayView, mut src_dst: Vec<(usize, usize)>) -> NdArray
{
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
fn transpose_reversed(perm: &NdArray) -> bool
{
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

pub fn logsumexp_forward(x: &NdArray, axis: isize) -> NdArray
{
    let axis = if axis < 0 {
        (x.ndim() as isize + axis) as usize
    } else {
        axis as usize
    };

    let mut a = x.shape().to_vec();
    a[axis] = 1;
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


impl ops::Op for LogSumExp {
    fn name(&self) -> &str
    {
        "LogSumExp"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x = xs[0];
        Ok(logsumexp_forward(x, self.axis))
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        // let ref sum = ops::exp(output);
        // let ref exp = ops::exp(inputs[0]);
        // let gx = gy * exp / sum;
        let gx = ops::softmax(inputs[0], self.axis) * gy;
        vec![Some(gx)]
    }
}


impl ops::Op for Pow {
    fn name(&self) -> &str
    {
        "Pow"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x0: &NdArray = xs[0];
        let a = self.a;
        Ok(x0.map(move |x| x.powf(a)))
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let x = inputs[0];
        let gx = gy * self.a * ops::pow(x, self.a - 1.);
        vec![Some(gx)]
    }
}

impl ops::Op for Sqrt {
    fn name(&self) -> &str
    {
        "Sqrt"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x0: &NdArray = xs[0];
        Ok(x0.map(|a| a.sqrt()))
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let x = inputs[0];
        let ret = 0.5 * ops::pow(x, -0.5);
        vec![Some(gy * ret)]
    }
}

impl ops::Op for Log {
    fn name(&self) -> &str
    {
        "Log"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].map(move |a| a.log(self.a)))
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(gy / inputs[0])]
    }
}

impl ops::Op for Exp {
    fn name(&self) -> &str
    {
        "Exp"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].map(|a| a.exp()))
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(output * gy)]
    }
}

impl ops::Op for Atanh {
    fn name(&self) -> &str
    {
        "Atanh"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let x = inputs[0];
        let y = 1 / (1 - x * x);
        vec![Some(y * gy)]
    }

    fn compute(&self, xs: &[&::NdArray]) -> Result<::NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].map(|a| a.atanh()))
    }
}

impl ops::Op for Acosh {
    fn name(&self) -> &str
    {
        "Acosh"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let x = inputs[0];
        let y = -1 / ops::sqrt(&(x * x - 1));
        vec![Some(y * gy)]
    }

    fn compute(&self, xs: &[&::NdArray]) -> Result<::NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].map(|a| a.acosh()))
    }
}

impl ops::Op for Asinh {
    fn name(&self) -> &str
    {
        "Asinh"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let x = inputs[0];
        let y = 1 / ops::sqrt(&(x * x + 1));
        vec![Some(y * gy)]
    }

    fn compute(&self, xs: &[&::NdArray]) -> Result<::NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].map(|a| a.asinh()))
    }
}

impl ops::Op for Tanh {
    fn name(&self) -> &str
    {
        "Tanh"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].map(|a| a.tanh()))
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], y: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(gy - &(y * y * gy))]
    }
}

impl ops::Op for Cosh {
    fn name(&self) -> &str
    {
        "Cosh"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(ops::sinh(inputs[0]) * gy)]
    }

    fn compute(&self, xs: &[&::NdArray]) -> Result<::NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].map(|a| a.cosh()))
    }
}

impl ops::Op for Sinh {
    fn name(&self) -> &str
    {
        "Sinh"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(ops::cosh(inputs[0]) * gy)]
    }

    fn compute(&self, xs: &[&::NdArray]) -> Result<::NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].map(|a| a.sinh()))
    }
}

impl ops::Op for Atan {
    fn name(&self) -> &str
    {
        "Atan"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let x = inputs[0];
        let y = 1 / (1 + x * x);
        vec![Some(y * gy)]
    }

    fn compute(&self, xs: &[&::NdArray]) -> Result<::NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].map(|a| a.atan()))
    }
}

impl ops::Op for Acos {
    fn name(&self) -> &str
    {
        "Acos"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let x = inputs[0];
        let y = -1 / ops::sqrt(&(1 - x * x));
        vec![Some(y * gy)]
    }

    fn compute(&self, xs: &[&::NdArray]) -> Result<::NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].map(|a| a.acos()))
    }
}

impl ops::Op for Asin {
    fn name(&self) -> &str
    {
        "Asin"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let x = inputs[0];
        let y = 1 / ops::sqrt(&(1 - x * x));
        vec![Some(y * gy)]
    }

    fn compute(&self, xs: &[&::NdArray]) -> Result<::NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].map(|a| a.asin()))
    }
}

impl ops::Op for Sin {
    fn name(&self) -> &str
    {
        "Sin"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(ops::cos(inputs[0]) * gy)]
    }

    fn compute(&self, xs: &[&::NdArray]) -> Result<::NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].map(|a| a.sin()))
    }
}

impl ops::Op for Cos {
    fn name(&self) -> &str
    {
        "Cos"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(-1 * ops::sin(inputs[0]) * gy)]
    }

    fn compute(&self, xs: &[&::NdArray]) -> Result<::NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].map(|a| a.cos()))
    }
}

impl ops::Op for Tan {
    fn name(&self) -> &str
    {
        "Tan"
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let ref cos = ops::cos(inputs[0]);
        vec![Some(gy / (cos * cos))]
    }

    fn compute(&self, xs: &[&::NdArray]) -> Result<::NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0].map(|a| a.tan()))
    }
}
