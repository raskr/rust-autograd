use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct Clip {
    pub min: f32,
    pub max: f32,
}

pub struct ClipGrad {
    pub min: f32,
    pub max: f32,
}

impl ops::Op for Clip {
    fn name(&self) -> &str
    {
        "Clip"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        xs[0].mapv(move |a| a.min(self.max).max(self.min))
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let op = ops::apply_op(
            ClipGrad {
                min: self.min,
                max: self.max,
            },
            &[inputs[0], gy],
        );
        vec![Some(op)]
    }
}

impl ops::Op for ClipGrad {
    fn name(&self) -> &str
    {
        "ClipGrad"
    }
    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let mut ret = xs[0].mapv(move |x| {
            // x > min && x < max
            (((x > self.min) as i32) as f32) * (((x < self.max) as i32) as f32)
        });
        ret *= xs[1];
        ret
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None]
    }
}
