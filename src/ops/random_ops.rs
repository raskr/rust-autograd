extern crate ndarray;

use ndarray_ext;
use op;
use tensor::Tensor;

pub struct StandardNormal;

pub struct StandardUniform;

pub struct RandomUniform {
    pub max: f64,
    pub min: f64,
}

pub struct RandomNormal {
    pub mean: f64,
    pub stddev: f64,
}

pub struct Bernoulli {
    pub p: f64,
}

pub struct Exponential {
    pub lambda: f64,
}

pub struct LogNormal {
    pub mean: f64,
    pub stddev: f64,
}

pub struct Gamma {
    pub shape_param: f64,
    pub scale: f64,
}


impl op::Op for RandomNormal {
    fn name(&self) -> &str
    {
        "RandomNormal"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![
            Ok(::ndarray_ext::random_normal(
                shape.as_slice(),
                self.mean,
                self.stddev,
            )),
        ]
    }
}

impl op::Op for RandomUniform {
    fn name(&self) -> &str
    {
        "RandomUniform"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![
            Ok(::ndarray_ext::random_uniform(
                shape.as_slice(),
                self.min,
                self.max,
            )),
        ]
    }
}

impl op::Op for StandardNormal {
    fn name(&self) -> &str
    {
        "StandardNormal"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![Ok(::ndarray_ext::standard_normal(shape.as_slice()))]
    }
}

impl op::Op for StandardUniform {
    fn name(&self) -> &str
    {
        "StandardUniform"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![Ok(::ndarray_ext::standard_uniform(shape.as_slice()))]
    }
}

impl op::Op for Bernoulli {
    fn name(&self) -> &str
    {
        "Bernoulli"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![Ok(::ndarray_ext::bernoulli(shape.as_slice(), self.p))]
    }
}

impl op::Op for Exponential {
    fn name(&self) -> &str
    {
        "Exponential"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![
            Ok(::ndarray_ext::exponential(shape.as_slice(), self.lambda)),
        ]
    }
}

impl op::Op for LogNormal {
    fn name(&self) -> &str
    {
        "LogNormal"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![
            Ok(::ndarray_ext::log_normal(
                shape.as_slice(),
                self.mean,
                self.stddev,
            )),
        ]
    }
}

impl op::Op for Gamma {
    fn name(&self) -> &str
    {
        "Gamma"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let shape = ndarray_ext::arr_to_shape(xs[0]);
        vec![
            Ok(::ndarray_ext::gamma(
                shape.as_slice(),
                self.shape_param,
                self.scale,
            )),
        ]
    }
}
