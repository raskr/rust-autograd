use op;
use tensor::Tensor;
use Float;

struct SGDOp<T: Float> {
    pub lr: T,
}

impl<T: Float> ::op::Op<T> for SGDOp<T> {
    fn name(&self) -> &str {
        "SGD"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResults<T> {
        let xs = ctx.grab_inputs();
        let grad = &xs[1];
        unsafe {
            ::ndarray_ext::axpy(&xs[0], -self.lr, grad.as_ptr(), grad.shape());
        }
        vec![Err(::op::ComputeException::NoOutput)]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

/// Vanilla SGD optimizer
pub struct SGD<T: Float> {
    pub lr: T,
}

impl<'a, T: Float> SGD<T> {
    pub fn compute_updates<A: AsRef<Tensor<T>>>(
        &self,
        params: &[&'a Tensor<T>],
        grads: &[A],
    ) -> Vec<Tensor<T>> {
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
