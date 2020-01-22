use crate::ops::gradient_descent_ops::sgd;
use crate::tensor::{Input, Tensor};
use crate::Float;
use crate::Graph;

/// Vanilla SGD optimizer
///
/// ```
/// extern crate autograd as ag;
///
/// let sgd = ag::optimizers::sgd::SGD { lr: 0.1 };
/// // let update_ops = sgd.compute_updates(params, grads)
/// ```
///
/// See also https://github.com/raskr/rust-autograd/blob/master/examples/mlp_mnist.rs
pub struct SGD<T: Float> {
    /// Learning rate
    pub lr: T,
}

impl<'a, 'b: 'a, T: Float> SGD<T> {
    /// Creates ops to optimize `params` with SGD.
    ///
    /// Evaluated results of the return values will be `None`.
    pub fn compute_updates(
        &self,
        params: Vec<Tensor<'a, 'b, T>>,
        grads: Vec<Tensor<'a, 'b, T>>,
        c: &'b Graph<T>,
    ) -> Vec<Tensor<'a, 'b, T>> {
        let len = params.len();
        let mut ret = Vec::with_capacity(len);
        for i in 0..len {
            ret.push(
                Tensor::builder()
                    .set_inputs_raw(vec![Input::new_mut(&params[i]), Input::new(&grads[i])])
                    .build(c, sgd::SGDOp::new(self.lr)),
            );
        }
        ret
    }
}
