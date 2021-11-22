//! Adagrad optimizer

use crate::optimizers::Optimizer;
use crate::tensor::Tensor;
use crate::tensor_ops::gradient_descent_ops::adagrad;
use crate::variable::VariableID;
use crate::{Context, Float, VariableEnvironment};

/// Adagrad optimizer
///
/// https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
pub struct AdaGrad<F: Float> {
    /// default: 0.01
    pub lr: F,
    pub adagrad_namespace_id: &'static str,
}

impl<'t, 'g, F: Float> AdaGrad<F> {
    pub fn default(
        adagrad_namespace_id: &'static str,
        var_id_list: impl IntoIterator<Item = VariableID>,
        env: &mut VariableEnvironment<F>,
    ) -> AdaGrad<F> {
        Self::new(
            F::from(0.01).unwrap(),
            var_id_list,
            env,
            adagrad_namespace_id,
        )
    }

    pub fn new(
        lr: F,
        var_id_list: impl IntoIterator<Item = VariableID>,
        env: &mut VariableEnvironment<F>,
        adagrad_namespace_id: &'static str,
    ) -> AdaGrad<F> {
        for vid in var_id_list.into_iter() {
            let h = {
                let target_var = env
                    .get_array_by_id(vid)
                    .expect("variable array not found")
                    .borrow();
                let var_shape = target_var.shape();
                crate::ndarray_ext::zeros(var_shape)
            };
            let mut ns = env.namespace_mut(adagrad_namespace_id);
            ns.slot().name(format!("{}", vid)).set(h);
        }
        AdaGrad {
            lr,
            adagrad_namespace_id,
        }
    }
}

impl<F: Float> Optimizer<F> for AdaGrad<F> {
    fn compute_updates<'g, A, B>(
        &self,
        params: &[A],
        grads: &[B],
        g: &'g Context<F>,
    ) -> Vec<Tensor<'g, F>>
    where
        A: AsRef<Tensor<'g, F>> + Copy,
        B: AsRef<Tensor<'g, F>> + Copy,
    {
        let num_params = params.len();
        assert_eq!(num_params, grads.len());
        let mut ret = Vec::with_capacity(num_params);
        for i in 0..num_params {
            let param = params[i].as_ref();
            let param_id = param.get_variable_id().expect("Got non-variable tensor");
            let ns = g.namespace(self.adagrad_namespace_id);
            let h = g.variable_by_name(format!("{}", param_id), &ns);

            ret.push(
                Tensor::builder(g)
                    .append_input(param, true)
                    .append_input(grads[i].as_ref(), false)
                    .append_input(&h, true)
                    .build(adagrad::AdaGradOp { lr: self.lr }),
            );
        }
        ret
    }
}
