use ndarray_ext::{self, ArrRng};
use op;
use rand::Rng;
use tensor::Tensor;
use Float;

pub struct StandardNormal<T: Float, R> {
    pub arr_rng: ArrRng<T, R>,
}

impl<T: Float, R> StandardNormal<T, R> {
    pub fn new(arr_rng: ArrRng<T, R>) -> Self {
        Self { arr_rng }
    }
}

pub struct StandardUniform<T: Float, R> {
    pub arr_rng: ArrRng<T, R>,
}

impl<T: Float, R> StandardUniform<T, R> {
    pub fn new(arr_rng: ArrRng<T, R>) -> Self {
        Self { arr_rng }
    }
}

pub struct RandomUniform<T: Float, R> {
    pub arr_rng: ArrRng<T, R>,
    pub max: f64,
    pub min: f64,
}

impl<T: Float, R> RandomUniform<T, R> {
    pub fn new(arr_rng: ArrRng<T, R>, min: f64, max: f64) -> Self {
        Self { arr_rng, max, min }
    }
}

pub struct RandomNormal<T: Float, R> {
    pub arr_rng: ArrRng<T, R>,
    pub mean: f64,
    pub stddev: f64,
}

impl<T: Float, R> RandomNormal<T, R> {
    pub fn new(arr_rng: ArrRng<T, R>, mean: f64, stddev: f64) -> Self {
        Self {
            arr_rng,
            mean,
            stddev,
        }
    }
}

pub struct Bernoulli<T: Float, R> {
    pub arr_rng: ArrRng<T, R>,
    pub p: f64,
}

impl<T: Float, R> Bernoulli<T, R> {
    pub fn new(arr_rng: ArrRng<T, R>, p: f64) -> Self {
        Self { arr_rng, p }
    }
}

pub struct Exponential<T: Float, R> {
    pub arr_rng: ArrRng<T, R>,
    pub lambda: f64,
}

impl<T: Float, R> Exponential<T, R> {
    pub fn new(arr_rng: ArrRng<T, R>, lambda: f64) -> Self {
        Self { arr_rng, lambda }
    }
}

pub struct LogNormal<T: Float, R> {
    pub arr_rng: ArrRng<T, R>,
    pub mean: f64,
    pub stddev: f64,
}

impl<T: Float, R> LogNormal<T, R> {
    pub fn new(arr_rng: ArrRng<T, R>, mean: f64, stddev: f64) -> Self {
        Self {
            arr_rng,
            mean,
            stddev,
        }
    }
}

pub struct Gamma<T: Float, R> {
    pub arr_rng: ArrRng<T, R>,
    pub shape_param: f64,
    pub scale: f64,
}

impl<T: Float, R> Gamma<T, R> {
    pub fn new(arr_rng: ArrRng<T, R>, shape_param: f64, scale: f64) -> Self {
        Self {
            arr_rng,
            shape_param,
            scale,
        }
    }
}

impl<T: Float, R: Rng> op::Op<T> for RandomNormal<T, R> {
    fn name(&self) -> &str {
        "RandomNormal"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![Ok(self.arr_rng.random_normal(
            shape.as_slice(),
            self.mean,
            self.stddev,
        ))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<R: Rng, T: Float> op::Op<T> for RandomUniform<T, R> {
    fn name(&self) -> &str {
        "RandomUniform"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![Ok(self.arr_rng.random_uniform(
            shape.as_slice(),
            self.min,
            self.max,
        ))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<R: Rng, T: Float> op::Op<T> for StandardNormal<T, R> {
    fn name(&self) -> &str {
        "StandardNormal"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![Ok(self.arr_rng.standard_normal(shape.as_slice()))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<R: Rng, T: Float> op::Op<T> for StandardUniform<T, R> {
    fn name(&self) -> &str {
        "StandardUniform"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![Ok(self.arr_rng.standard_uniform(shape.as_slice()))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<R: Rng, T: Float> op::Op<T> for Bernoulli<T, R> {
    fn name(&self) -> &str {
        "Bernoulli"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![Ok(self.arr_rng.bernoulli(shape.as_slice(), self.p))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<R: Rng, T: Float> op::Op<T> for Exponential<T, R> {
    fn name(&self) -> &str {
        "Exponential"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![Ok(self.arr_rng.exponential(shape.as_slice(), self.lambda))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<R: Rng, T: Float> op::Op<T> for LogNormal<T, R> {
    fn name(&self) -> &str {
        "LogNormal"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![Ok(self.arr_rng.log_normal(
            shape.as_slice(),
            self.mean,
            self.stddev,
        ))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<R: Rng, T: Float> op::Op<T> for Gamma<T, R> {
    fn name(&self) -> &str {
        "Gamma"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![Ok(self.arr_rng.gamma(
            shape.as_slice(),
            self.shape_param,
            self.scale,
        ))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}
