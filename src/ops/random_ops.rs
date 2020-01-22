use crate::ndarray_ext::{self, ArrRng};
use crate::op;
use crate::Float;
use rand::Rng;

pub struct StandardNormal<T: Float, R: Rng + Send> {
    pub arr_rng: ArrRng<T, R>,
}

impl<'a, T: Float, R: Rng + Send> StandardNormal<T, R> {
    pub fn new(arr_rng: ArrRng<T, R>) -> Self {
        Self { arr_rng }
    }
}

pub struct StandardUniform<T: Float, R: Rng + Send> {
    pub arr_rng: ArrRng<T, R>,
}

impl<'a, T: Float, R: Rng + Send> StandardUniform<T, R> {
    pub fn new(arr_rng: ArrRng<T, R>) -> Self {
        Self { arr_rng }
    }
}

pub struct RandomUniform<T: Float, R: Rng + Send> {
    pub arr_rng: ArrRng<T, R>,
    pub max: f64,
    pub min: f64,
}

impl<'a, T: Float, R: Rng + Send> RandomUniform<T, R> {
    pub fn new(arr_rng: ArrRng<T, R>, min: f64, max: f64) -> Self {
        Self { arr_rng, max, min }
    }
}

pub struct RandomNormal<T: Float, R: Rng + Send> {
    pub arr_rng: ArrRng<T, R>,
    pub mean: f64,
    pub stddev: f64,
}

impl<'a, T: Float, R: Rng + Send> RandomNormal<T, R> {
    pub fn new(arr_rng: ArrRng<T, R>, mean: f64, stddev: f64) -> Self {
        Self {
            arr_rng,
            mean,
            stddev,
        }
    }
}

pub struct Bernoulli<T: Float, R: Rng + Send> {
    pub arr_rng: ArrRng<T, R>,
    pub p: f64,
}

impl<'a, T: Float, R: Rng + Send> Bernoulli<T, R> {
    pub fn new(arr_rng: ArrRng<T, R>, p: f64) -> Self {
        Self { arr_rng, p }
    }
}

pub struct Exponential<T: Float, R: Rng + Send> {
    pub arr_rng: ArrRng<T, R>,
    pub lambda: f64,
}

impl<'a, T: Float, R: Rng + Send> Exponential<T, R> {
    pub fn new(arr_rng: ArrRng<T, R>, lambda: f64) -> Self {
        Self { arr_rng, lambda }
    }
}

pub struct LogNormal<T: Float, R: Rng + Send> {
    pub arr_rng: ArrRng<T, R>,
    pub mean: f64,
    pub stddev: f64,
}

impl<'a, T: Float, R: Rng + Send> LogNormal<T, R> {
    pub fn new(arr_rng: ArrRng<T, R>, mean: f64, stddev: f64) -> Self {
        Self {
            arr_rng,
            mean,
            stddev,
        }
    }
}

pub struct Gamma<T: Float, R: Rng + Send> {
    pub arr_rng: ArrRng<T, R>,
    pub shape_param: f64,
    pub scale: f64,
}

impl<'a, T: Float, R: Rng + Send> Gamma<T, R> {
    pub fn new(arr_rng: ArrRng<T, R>, shape_param: f64, scale: f64) -> Self {
        Self {
            arr_rng,
            shape_param,
            scale,
        }
    }
}

impl<T: Float, R: Rng + Send> op::Op<T> for RandomNormal<T, R> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let shape = ndarray_ext::as_shape(&ctx.input(0));
        ctx.append_output(Ok(crate::ArrRepr::Owned(self.arr_rng.random_normal(
            shape.as_slice(),
            self.mean,
            self.stddev,
        ))));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None]);
    }
}

impl<R: Rng + Send, T: Float> op::Op<T> for RandomUniform<T, R> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let shape = ndarray_ext::as_shape(&ctx.input(0));
        ctx.append_output(Ok(crate::ArrRepr::Owned(self.arr_rng.random_uniform(
            shape.as_slice(),
            self.min,
            self.max,
        ))));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None]);
    }
}

impl<R: Rng + Send, T: Float> op::Op<T> for StandardNormal<T, R> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let shape = ndarray_ext::as_shape(&ctx.input(0));
        ctx.append_output(Ok(crate::ArrRepr::Owned(
            self.arr_rng.standard_normal(shape.as_slice()),
        )));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None]);
    }
}

impl<R: Rng + Send, T: Float> op::Op<T> for StandardUniform<T, R> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let shape = ndarray_ext::as_shape(&ctx.input(0));
        ctx.append_output(Ok(crate::ArrRepr::Owned(
            self.arr_rng.standard_uniform(shape.as_slice()),
        )));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None]);
    }
}

impl<R: Rng + Send, T: Float> op::Op<T> for Bernoulli<T, R> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let shape = ndarray_ext::as_shape(&ctx.input(0));
        ctx.append_output(Ok(crate::ArrRepr::Owned(
            self.arr_rng.bernoulli(shape.as_slice(), self.p),
        )));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None]);
    }
}

impl<R: Rng + Send, T: Float> op::Op<T> for Exponential<T, R> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let shape = ndarray_ext::as_shape(&ctx.input(0));
        ctx.append_output(Ok(crate::ArrRepr::Owned(
            self.arr_rng.exponential(shape.as_slice(), self.lambda),
        )));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None]);
    }
}

impl<R: Rng + Send, T: Float> op::Op<T> for LogNormal<T, R> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let shape = ndarray_ext::as_shape(&ctx.input(0));
        ctx.append_output(Ok(crate::ArrRepr::Owned(self.arr_rng.log_normal(
            shape.as_slice(),
            self.mean,
            self.stddev,
        ))));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None]);
    }
}

impl<R: Rng + Send, T: Float> op::Op<T> for Gamma<T, R> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let shape = ndarray_ext::as_shape(&ctx.input(0));
        ctx.append_output(Ok(crate::ArrRepr::Owned(self.arr_rng.gamma(
            shape.as_slice(),
            self.shape_param,
            self.scale,
        ))));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None]);
    }
}
