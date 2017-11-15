use ops;
use tensor::Tensor;

pub struct DummyOp {
    pub name: String,
}

impl ops::Op for DummyOp {
    fn name(&self) -> &str
    {
        &self.name
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        panic!(
            "must not be called ({}#grad). This is probably bug.",
            self.name
        )
    }

    fn compute(&self, _: &[&::NdArray], _: bool) -> ::NdArray
    {
        let msg = if self.name == "Placeholder" {
            "There exists placeholder(s) couldn't get initial value"
        } else if self.name == "Variable" {
            "Current graph evaluation context doesn't match with what generated this variable."
        } else {
            unreachable!()
        };
        panic!(msg);
    }
}
