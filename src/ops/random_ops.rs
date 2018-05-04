extern crate ndarray;

use ndarray_ext::{self, ArrRng};
use op;
use tensor::Tensor;
use rand::Rng;

pub struct StandardNormal<R>
{
    pub arr_rng: ArrRng<R>,
}

impl<R> StandardNormal<R> {
    pub fn new(arr_rng: ArrRng<R>) -> Self {
        Self {
            arr_rng,
        }
    }
}

pub struct StandardUniform<R>
{
    pub arr_rng: ArrRng<R>,
}

impl<R> StandardUniform<R> {
    pub fn new(arr_rng: ArrRng<R>) -> Self {
        Self { arr_rng }
    }
}

pub struct RandomUniform<R>
{
    pub arr_rng: ArrRng<R>,
    pub max: f64,
    pub min: f64,
}

impl<R> RandomUniform<R> {
    pub fn new(arr_rng: ArrRng<R>, min: f64, max: f64) -> Self {
        Self { arr_rng, max, min }
    }
}

pub struct RandomNormal<R>
{
    pub arr_rng: ArrRng<R>,
    pub mean:   f64,
    pub stddev: f64,
}

impl<R> RandomNormal<R> {
    pub fn new(arr_rng: ArrRng<R>, mean: f64, stddev: f64) -> Self {
        Self { arr_rng, mean, stddev }
    }
}

pub struct Bernoulli<R>
{
    pub arr_rng: ArrRng<R>,
    pub p: f64,
}

impl<R> Bernoulli<R> {
    pub fn new(arr_rng: ArrRng<R>, p: f64) -> Self {
        Self { arr_rng, p }
    }
}

pub struct Exponential<R>
{
    pub arr_rng: ArrRng<R>,
    pub lambda: f64,
}

impl<R> Exponential<R> {
    pub fn new(arr_rng: ArrRng<R>, lambda: f64) -> Self {
        Self { arr_rng, lambda }
    }
}

pub struct LogNormal<R>
{
    pub arr_rng: ArrRng<R>,
    pub mean:   f64,
    pub stddev: f64,
}

impl<R> LogNormal<R> {
    pub fn new(arr_rng: ArrRng<R>, mean: f64, stddev: f64) -> Self {
        Self { arr_rng, mean, stddev }
    }
}

pub struct Gamma<R>
{
    pub arr_rng: ArrRng<R>,
    pub shape_param: f64,
    pub scale:       f64,
}

impl<R> Gamma<R> {
    pub fn new(arr_rng: ArrRng<R>, shape_param: f64, scale: f64) -> Self {
        Self { arr_rng, shape_param, scale }
    }
}

impl<R: Rng> op::Op for RandomNormal<R>
{
    fn name(&self) -> &str
    {
        "RandomNormal"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![
            Ok(self.arr_rng.random_normal(
                shape.as_slice(),
                self.mean,
                self.stddev,
            )),
        ]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

impl<R: Rng> op::Op for RandomUniform<R>
{
    fn name(&self) -> &str
    {
        "RandomUniform"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![
            Ok(self.arr_rng.random_uniform(
                shape.as_slice(),
                self.min,
                self.max,
            )),
        ]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

impl<R: Rng> op::Op for StandardNormal<R>
{
    fn name(&self) -> &str
    {
        "StandardNormal"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![Ok(self.arr_rng.standard_normal(shape.as_slice()))]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

impl<R: Rng> op::Op for StandardUniform<R>
{
    fn name(&self) -> &str
    {
        "StandardUniform"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![Ok(self.arr_rng.standard_uniform(shape.as_slice()))]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

impl<R: Rng> op::Op for Bernoulli<R>
{
    fn name(&self) -> &str
    {
        "Bernoulli"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![Ok(self.arr_rng.bernoulli(shape.as_slice(), self.p))]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

impl<R: Rng> op::Op for Exponential<R>
{
    fn name(&self) -> &str
    {
        "Exponential"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![
            Ok(self.arr_rng.exponential(shape.as_slice(), self.lambda)),
        ]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

impl<R: Rng> op::Op for LogNormal<R>
{
    fn name(&self) -> &str
    {
        "LogNormal"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![
            Ok(self.arr_rng.log_normal(
                shape.as_slice(),
                self.mean,
                self.stddev,
            )),
        ]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

impl<R: Rng> op::Op for Gamma<R>
{
    fn name(&self) -> &str
    {
        "Gamma"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![
            Ok(self.arr_rng.gamma(
                shape.as_slice(),
                self.shape_param,
                self.scale,
            )),
        ]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}
