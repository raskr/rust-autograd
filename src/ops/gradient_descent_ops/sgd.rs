use ndarray_ext::NdArray;
use tensor::Tensor;


struct SGDOp {
    pub lr: f32,
}

impl ::ops::Op for SGDOp {
    fn name(&self) -> &str
    {
        "SGD"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let mut xs = unsafe { ctx.grab_assignable_inputs() };
        let updates = {
            let grad: &NdArray = xs[1];
            grad * self.lr
        };
        xs[0].zip_mut_with(&updates, |a, &b| *a -= b);
        Err(::ops::OpComputeErrorStatus::NoOutput)
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}


/// Vanilla SGD optimizer
pub struct SGD {
    pub lr: f32,
}

impl<'a> super::Optimizer<'a> for SGD {
    fn compute_updates<T: AsRef<Tensor>>(
        &mut self,
        params: &[&'a Tensor],
        grads: &[T],
    ) -> Vec<Tensor>
    {
        params
            .into_iter()
            .zip(grads)
            .map(|(param, grad)| {
                ::ops::apply_op(SGDOp { lr: self.lr }, &[param, grad.as_ref()], None)
            })
            .collect()
    }
}
