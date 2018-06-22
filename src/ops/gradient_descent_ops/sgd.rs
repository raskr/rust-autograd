use ndarray_ext::NdArray;
use op;
use tensor::Tensor;

struct SGDOp
{
    pub lr: f32,
}

impl ::op::Op for SGDOp
{
    fn name(&self) -> &str
    {
        "SGD"
    }

    fn compute(&self, mut ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = unsafe { ctx.grab_assignable_inputs() };
        let updates = {
            let grad: &NdArray = xs[1];
            grad * self.lr
        };
        xs[0].zip_mut_with(&updates, |a, &b| *a -= b);
        vec![Err(::op::ComputeError::NoOutput)]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

/// Vanilla SGD optimizer
pub struct SGD
{
    pub lr: f32,
}

impl<'a> SGD
{
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
                Tensor::builder()
                    .set_inputs(vec![param, grad.as_ref()])
                    .build(SGDOp { lr: self.lr })
            })
            .collect()
    }
}
