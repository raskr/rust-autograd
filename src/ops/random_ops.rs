use ops;
use tensor::Tensor;


pub struct RandomNormal {
    pub shape: Vec<usize>,
    pub mean: f64,
    pub stddev: f64,
}

pub struct RandomUniform {
    pub shape: Vec<usize>,
    pub max: f64,
    pub min: f64,
}

pub struct StandardNormal {
    pub shape: Vec<usize>,
}

pub struct StandardUniform {
    pub shape: Vec<usize>,
}

pub struct Bernoulli {
    pub shape: Vec<usize>,
    pub p: f64,
}

pub struct Exponential {
    pub shape: Vec<usize>,
    pub lambda: f64,
}

pub struct LogNormal {
    pub shape: Vec<usize>,
    pub mean: f64,
    pub stddev: f64,
}

pub struct Gamma {
    pub shape: Vec<usize>,
    pub shape_param: f64,
    pub scale: f64,
}


impl ops::Op for RandomNormal {
    fn name(&self) -> &str
    {
        "RandomNormal"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, _: &[&::NdArray], _: bool) -> ::NdArray
    {
        ::ndarray_ext::random_normal(self.shape.as_slice(), self.mean, self.stddev)
    }
}

impl ops::Op for RandomUniform {
    fn name(&self) -> &str
    {
        "RandomUniform"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, _: &[&::NdArray], _: bool) -> ::NdArray
    {
        ::ndarray_ext::random_uniform(self.shape.as_slice(), self.min, self.max)
    }
}

impl ops::Op for StandardNormal {
    fn name(&self) -> &str
    {
        "StandardNormal"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, _: &[&::NdArray], _: bool) -> ::NdArray
    {
        ::ndarray_ext::standard_normal(self.shape.as_slice())
    }
}

impl ops::Op for StandardUniform {
    fn name(&self) -> &str
    {
        "StandardUniform"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, _: &[&::NdArray], _: bool) -> ::NdArray
    {
        ::ndarray_ext::standard_uniform(self.shape.as_slice())
    }
}

impl ops::Op for Bernoulli {
    fn name(&self) -> &str
    {
        "Bernoulli"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, _: &[&::NdArray], _: bool) -> ::NdArray
    {
        ::ndarray_ext::bernoulli(self.shape.as_slice(), self.p)
    }
}

impl ops::Op for Exponential {
    fn name(&self) -> &str
    {
        "Exponential"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, _: &[&::NdArray], _: bool) -> ::NdArray
    {
        ::ndarray_ext::exponential(self.shape.as_slice(), self.lambda)
    }
}

impl ops::Op for LogNormal {
    fn name(&self) -> &str
    {
        "LogNormal"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, _: &[&::NdArray], _: bool) -> ::NdArray
    {
        ::ndarray_ext::log_normal(self.shape.as_slice(), self.mean, self.stddev)
    }
}

impl ops::Op for Gamma {
    fn name(&self) -> &str
    {
        "Gamma"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, _: &[&::NdArray], _: bool) -> ::NdArray
    {
        ::ndarray_ext::gamma(self.shape.as_slice(), self.shape_param, self.scale)
    }
}
