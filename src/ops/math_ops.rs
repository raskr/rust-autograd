use ndarray_ext::NdArray;
use ops;
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
pub struct Log {
    pub a: f32,
}
pub struct Pow {
    pub a: f32,
}


impl ops::Op for Pow {
    fn name(&self) -> &str
    {
        "Pow"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x0: &NdArray = xs[0];
        x0.map(move |a| a.powf(self.a))
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

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x0: &NdArray = xs[0];
        x0.map(|a| a.sqrt())
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

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        xs[0].map(move |a| a.log(self.a))
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

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        xs[0].map(|a| a.exp())
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

    fn compute(&self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        xs[0].map(|a| a.atanh())
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

    fn compute(&self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        xs[0].map(|a| a.acosh())
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

    fn compute(&self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        xs[0].map(|a| a.asinh())
    }
}

impl ops::Op for Tanh {
    fn name(&self) -> &str
    {
        "Tanh"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        xs[0].map(|a| a.tanh())
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some((1. - output * output) * gy)]
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

    fn compute(&self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        xs[0].map(|a| a.cosh())
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

    fn compute(&self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        xs[0].map(|a| a.sinh())
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

    fn compute(&self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        xs[0].map(|a| a.atan())
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

    fn compute(&self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        xs[0].map(|a| a.acos())
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

    fn compute(&self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        xs[0].map(|a| a.asin())
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

    fn compute(&self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        xs[0].map(|a| a.sin())
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

    fn compute(&self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        xs[0].map(|a| a.cos())
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

    fn compute(&self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        xs[0].map(|a| a.tan())
    }
}
