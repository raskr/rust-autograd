use ndarray_ext::NdArray;
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

    fn compute(&self, mut ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = unsafe { ctx.grab_assignable_inputs() };
        let updates = {
            let grad: &NdArray<T> = xs[1];
            grad.mapv(|x| x * self.lr)
        };
        xs[0].zip_mut_with(&updates, |a, &b| *a -= b);
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
        &mut self,
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
