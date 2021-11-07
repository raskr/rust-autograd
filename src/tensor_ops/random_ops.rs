use crate::ndarray_ext::{self, ArrayRng};
use crate::op;
use crate::Float;
use rand::Rng;

pub struct StandardNormal<T: Float, R: Rng> {
    pub arr_rng: ArrayRng<T, R>,
}

impl<'a, T: Float, R: Rng> StandardNormal<T, R> {
    pub fn new(arr_rng: ArrayRng<T, R>) -> Self {
        Self { arr_rng }
    }
}

pub struct StandardUniform<T: Float, R: Rng> {
    pub arr_rng: ArrayRng<T, R>,
}

impl<'a, T: Float, R: Rng> StandardUniform<T, R> {
    pub fn new(arr_rng: ArrayRng<T, R>) -> Self {
        Self { arr_rng }
    }
}

pub struct RandomUniform<T: Float, R: Rng> {
    pub arr_rng: ArrayRng<T, R>,
    pub max: f64,
    pub min: f64,
}

impl<'a, T: Float, R: Rng> RandomUniform<T, R> {
    pub fn new(arr_rng: ArrayRng<T, R>, min: f64, max: f64) -> Self {
        Self { arr_rng, max, min }
    }
}

pub struct RandomNormal<T: Float, R: Rng> {
    pub arr_rng: ArrayRng<T, R>,
    pub mean: f64,
    pub stddev: f64,
}

impl<'a, T: Float, R: Rng> RandomNormal<T, R> {
    pub fn new(arr_rng: ArrayRng<T, R>, mean: f64, stddev: f64) -> Self {
        Self {
            arr_rng,
            mean,
            stddev,
        }
    }
}

pub struct Bernoulli<T: Float, R: Rng> {
    pub arr_rng: ArrayRng<T, R>,
    pub p: f64,
}

impl<'a, T: Float, R: Rng> Bernoulli<T, R> {
    pub fn new(arr_rng: ArrayRng<T, R>, p: f64) -> Self {
        Self { arr_rng, p }
    }
}

pub struct Exponential<T: Float, R: Rng> {
    pub arr_rng: ArrayRng<T, R>,
    pub lambda: f64,
}

impl<'a, T: Float, R: Rng> Exponential<T, R> {
    pub fn new(arr_rng: ArrayRng<T, R>, lambda: f64) -> Self {
        Self { arr_rng, lambda }
    }
}

pub struct LogNormal<T: Float, R: Rng> {
    pub arr_rng: ArrayRng<T, R>,
    pub mean: f64,
    pub stddev: f64,
}

impl<'a, T: Float, R: Rng> LogNormal<T, R> {
    pub fn new(arr_rng: ArrayRng<T, R>, mean: f64, stddev: f64) -> Self {
        Self {
            arr_rng,
            mean,
            stddev,
        }
    }
}

pub struct Gamma<T: Float, R: Rng> {
    pub arr_rng: ArrayRng<T, R>,
    pub shape_param: f64,
    pub scale: f64,
}

impl<'a, T: Float, R: Rng> Gamma<T, R> {
    pub fn new(arr_rng: ArrayRng<T, R>, shape_param: f64, scale: f64) -> Self {
        Self {
            arr_rng,
            shape_param,
            scale,
        }
    }
}

impl<T: Float, R: Rng> op::Op<T> for RandomNormal<T, R> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::as_shape(&ctx.input(0));
        ctx.append_output(
            self.arr_rng
                .random_normal(shape.as_slice(), self.mean, self.stddev),
        );
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None);
    }
}

impl<R: Rng, T: Float> op::Op<T> for RandomUniform<T, R> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::as_shape(&ctx.input(0));
        ctx.append_output(
            self.arr_rng
                .random_uniform(shape.as_slice(), self.min, self.max),
        );
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None);
    }
}

impl<R: Rng, T: Float> op::Op<T> for StandardNormal<T, R> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::as_shape(&ctx.input(0));
        ctx.append_output(self.arr_rng.standard_normal(shape.as_slice()));
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None);
    }
}

impl<R: Rng, T: Float> op::Op<T> for StandardUniform<T, R> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::as_shape(&ctx.input(0));
        ctx.append_output(self.arr_rng.standard_uniform(shape.as_slice()));
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None);
    }
}

impl<R: Rng, T: Float> op::Op<T> for Bernoulli<T, R> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::as_shape(&ctx.input(0));
        ctx.append_output(self.arr_rng.bernoulli(shape.as_slice(), self.p));
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None);
    }
}

impl<R: Rng, T: Float> op::Op<T> for Exponential<T, R> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::as_shape(&ctx.input(0));
        ctx.append_output(self.arr_rng.exponential(shape.as_slice(), self.lambda));
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None);
    }
}

impl<R: Rng, T: Float> op::Op<T> for LogNormal<T, R> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::as_shape(&ctx.input(0));
        ctx.append_output(
            self.arr_rng
                .log_normal(shape.as_slice(), self.mean, self.stddev),
        );
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None);
    }
}

impl<R: Rng, T: Float> op::Op<T> for Gamma<T, R> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::as_shape(&ctx.input(0));
        ctx.append_output(
            self.arr_rng
                .gamma(shape.as_slice(), self.shape_param, self.scale),
        );
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None);
    }
}
