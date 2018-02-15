extern crate ndarray;

use ndarray_ext::NdArray;
use op;
use ops;
use std::f32;
use tensor::Tensor;


pub struct SoftmaxCrossEntropy;
pub struct SparseSoftmaxCrossEntropy;
pub struct SparseSoftmaxCrossEntropyGrad;
pub struct SigmoidCrossEntropy;
pub struct LogSoftmax {
    pub axis: isize,
}


impl op::Op for LogSoftmax {
    fn name(&self) -> &str
    {
        "LogSoftmax"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        vec![
            Ok(xs[0] - &ops::math_ops::logsumexp_forward(xs[0], self.axis, true)),
        ]
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        let sm = ops::exp(output);
        let sum = ops::reduce_sum(gy, &[1], true);
        let ref mul = sm * sum;
        vec![Some(gy - mul)]
    }
}

impl op::Op for SigmoidCrossEntropy {
    fn name(&self) -> &str
    {
        "SigmoidCrossEntropy"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let t = xs[1];

        if x.shape() != t.shape() {
            return vec![
                Err(::OpComputeErrorStatus::BadInput(
                    "x.shape must match t.shape".to_string(),
                )),
            ];
        }

        let e = f32::consts::E;
        let max_fn = f32::max;
        let mut tmp = x.map(move |a| ((-a.abs()).exp() + 1.).log(e) + max_fn(0., *a));
        tmp -= &(t * x);
        vec![Ok(tmp)]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let x = inputs[0];
        let t = inputs[1];

        let gx1 = {
            let ref exp = ops::exp(x);
            ((exp / (exp + 1)) - t) * gy
        };

        let gx2 = ops::neg(&(gy * t));

        vec![Some(gx1), Some(gx2)]
    }
}

impl op::Op for SparseSoftmaxCrossEntropy {
    fn name(&self) -> &str
    {
        "SparseSoftmaxCrossEntropy"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let (x, t) = (xs[0], xs[1]);
        let log_x: NdArray = x - &ops::math_ops::logsumexp_forward(x, 1, true);

        // validation
        {
            if log_x.ndim() != 2 {
                return vec![
                    Err(::OpComputeErrorStatus::BadInput(
                        format!("Bad first argument's shape {:?}", log_x.shape()),
                    )),
                ];
            }

            let t_shape = t.shape();
            let t_rank = t_shape.len();
            if t_rank == 2 {
                if t_shape[1] != 1 {
                    return vec![
                        Err(::OpComputeErrorStatus::BadInput(
                            format!("Bad second argument's shape {:?}", t_shape),
                        )),
                    ];
                }
            } else if t_rank != 1 {
                return vec![
                    Err(::OpComputeErrorStatus::BadInput(
                        format!("Bad second argument's shape {:?}", t_shape),
                    )),
                ];
            }
        }

        let mut t_iter = t.iter();

        // unwrap is safe
        let ret = log_x
            .map_axis(ndarray::Axis(1), move |row| {
                -row[*t_iter.next().unwrap() as usize]
            })
            .into_shape(ndarray::IxDyn(&[log_x.shape()[0], 1]))
            .unwrap();

        vec![Ok(ret), Ok(log_x)]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        let t = inputs[1];
        let ref log_x = ops::select_ith_of(output, 1);

        let gx1 = Tensor::builder().set_inputs(vec![log_x, t, gy]).build(
            SparseSoftmaxCrossEntropyGrad,
        );

        // gx2 won't be used
        let gx2 = {
            let ref x = ops::exp(log_x);
            let sum = ops::reduce_sum(&(x * log_x), &[1], true);
            x * gy * (sum - log_x)
        };

        vec![Some(gx1), Some(gx2)]
    }
}

impl op::Op for SparseSoftmaxCrossEntropyGrad {
    fn name(&self) -> &str
    {
        "SparseSoftmaxCrossEntropyGrad"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let log_x = xs[0]; // x is softmax  [2, 1]
        let t = xs[1]; // (2,)
        let gy = xs[2]; // (0)
        let mut x = log_x.map(|a| a.exp());

        for (mut row, &t_) in x.axis_iter_mut(ndarray::Axis(0)).zip(t) {
            row[t_ as usize] -= 1.;
        }

        x *= gy;
        vec![Ok(x)]

    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None]
    }
}


impl op::Op for SoftmaxCrossEntropy {
    fn name(&self) -> &str
    {
        "SoftmaxCrossEntropy"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let log_x: NdArray = x - &ops::math_ops::logsumexp_forward(x, 1, true);
        // `t` must be one-hot
        let t = xs[1];

        if log_x.ndim() != 2 {
            return vec![
                Err(::OpComputeErrorStatus::BadInput(
                    "x must be 2-ranked tensor".to_string(),
                )),
            ];
        }
        if t.ndim() != 2 {
            return vec![
                Err(::OpComputeErrorStatus::BadInput(
                    "t must be 2-ranked tensor".to_string(),
                )),
            ];
        }

        // - t log x ( =(batch, num_classes))
        vec![Ok((t * &log_x).sum_axis(ndarray::Axis(1)) * -1.), Ok(log_x)]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        let ref log_x = ops::select_ith_of(output, 1);
        let ref x = ops::exp(log_x);
        let t = inputs[1];

        // x = softmax, gy = dy/dx
        // = {gy - Σ(x * gy)} * x
        // = {-t/x - Σ(x * -t/x)} * x
        // = {-t/x + Σt} * x
        // = -t + x
        let gx1 = (x - t) * gy;

        // gx2 won't be used
        let gx2 = {
            let sum = ops::reduce_sum(&(x * log_x), &[-1], true);
            gy * (sum - log_x) * output
        };

        vec![Some(gx1), Some(gx2)]
    }
}
