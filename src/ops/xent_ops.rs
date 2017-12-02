extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use std::f32;
use tensor::Tensor;


pub struct SoftmaxCrossEntropyLatter;
pub struct SparseSoftmaxCrossEntropyLatter;
pub struct SparseSoftmaxCrossEntropyGrad;
pub struct SigmoidCrossEntropy;
pub struct LogSoftmax {
    pub axis: isize,
}


impl ops::Op for LogSoftmax {
    fn name(&self) -> &str
    {
        "LogSoftmax"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        Ok(xs[0] - &ops::math_ops::logsumexp_forward(xs[0], self.axis, true))
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        let sm = ops::exp(output);
        let sum = ops::reduce_sum(gy, &[1], true);
        let ref mul = sm * sum;
        vec![Some(gy - mul)]
    }
}

impl ops::Op for SigmoidCrossEntropy {
    fn name(&self) -> &str
    {
        "SigmoidCrossEntropy"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x = xs[0];
        let t = xs[1];

        if x.shape() != t.shape() {
            return Err(::OpComputeErrorStatus::BadInput(
                "x.shape must match t.shape".to_string(),
            ));
        }

        let e = f32::consts::E;
        let max_fn = f32::max;
        let mut tmp = x.map(move |a| ((-a.abs()).exp() + 1.).log(e) + max_fn(0., *a));
        tmp -= &(t * x);
        Ok(tmp)
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

impl ops::Op for SparseSoftmaxCrossEntropyLatter {
    fn name(&self) -> &str
    {
        "SparseSoftmaxCrossEntropyLatter"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let (log_x, t) = (xs[0], xs[1]);

        // validation
        {
            if log_x.ndim() != 2 {
                return Err(::OpComputeErrorStatus::BadInput(
                    format!("Bad first argument's shape {:?}", log_x.shape())
                ));
            }

            let t_shape = t.shape();
            let t_rank = t_shape.len();
            if t_rank == 2 {
                if t_shape[1] != 1 {
                    return Err(::OpComputeErrorStatus::BadInput(
                        format!("Bad second argument's shape {:?}", t_shape)
                    ));
                }
            } else if t_rank != 1 {
                return Err(::OpComputeErrorStatus::BadInput(
                    format!("Bad second argument's shape {:?}", t_shape)
                ));
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

        Ok(ret)
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let log_x = inputs[0];
        let t = inputs[1];

        let gx1 = ops::apply_op(
            SparseSoftmaxCrossEntropyGrad,
            &[log_x, t, gy],
            Some(log_x.shape()),
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

impl ops::Op for SparseSoftmaxCrossEntropyGrad {
    fn name(&self) -> &str
    {
        "SparseSoftmaxCrossEntropyGrad"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let log_x = xs[0]; // x is softmax
        let t = xs[1];
        let gy = xs[2];

        let mut x = log_x.map(|a| a.exp());

        for (mut row, &t_) in x.axis_iter_mut(ndarray::Axis(0)).zip(t) {
            row[t_ as usize] -= 1.;
        }

        x *= gy;
        Ok(x)

    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None]
    }
}


impl ops::Op for SoftmaxCrossEntropyLatter {
    fn name(&self) -> &str
    {
        "SoftmaxCrossEntropyLatter"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let log_x = xs[0];
        // `t` must be one-hot
        let t = xs[1];

        if log_x.ndim() != 2 {
            return Err(::OpComputeErrorStatus::BadInput(
                "x must be 2-ranked tensor".to_string(),
            ));
        }
        if t.ndim() != 2 {
            return Err(::OpComputeErrorStatus::BadInput(
                "t must be 2-ranked tensor".to_string(),
            ));
        }

        // - t log x ( =(batch, num_classes))
        Ok((t * log_x).sum_axis(ndarray::Axis(1)) * -1.)
    }

    fn grad(&self, output_grad: &Tensor, inputs: &[&Tensor], output: &Tensor)
        -> Vec<Option<Tensor>>
    {
        let ref x = ops::exp(inputs[0]);
        let t = inputs[1];

        // x = output of softmax, gy = dy/dx
        // = {gy - Σ(x * gy)} * x
        // = {-t/x - Σ(x * -t/x)} * x
        // = {-t/x + Σt} * x
        // = -t + x
        let gx1 = (x - t) * output_grad;

        // gx2 won't be used
        let gx2 = {
            let ref log_x = ops::log_softmax(inputs[0], -1);
            let sum = ops::reduce_sum(&(x * log_x), &[-1], true);
            output_grad * (sum - log_x) * output
        };

        vec![Some(gx1), Some(gx2)]
    }
}
