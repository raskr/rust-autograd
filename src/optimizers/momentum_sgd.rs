//! Momentum SGD optimizer
use crate::evaluation::Feeder;
use crate::optimizers::Optimizer;
use crate::tensor::Tensor;
use crate::tensor_ops::gradient_descent_ops::sgd;
use crate::variable::VariableID;
use crate::{Context, Float, NdArray, VariableEnvironment};

/// Momentum gradient descent optimizer
///
/// Use `ag::tensor_ops::gradient_descent` for the banilla sgd.
///
/// ```
/// use autograd as ag;
/// use ag::prelude::*;
/// use ag::optimizers;
/// use ag::optimizers::momentum_sgd::MomentumSGD;
///
/// type Tensor<'g> = ag::Tensor<'g, f64>;
/// let mut env = ag::VariableEnvironment::new();
/// let opt = MomentumSGD::default("sgd", env.default_namespace().current_var_ids(), &mut env);
///
/// env.run(|g| {
///    let p = g.placeholder("p", &[]);
///
///    let mut feeder = ag::Feeder::new();
///    let feed = ag::ndarray::arr0(2.);
///    feeder.push(p, feed.view());
///
///    let (params, grads): (&[Tensor], &[Tensor]) = (&[], &[]); // dummy
///    opt.update(params, grads, g, feeder); // do parameter update
/// });
/// ```
pub struct MomentumSGD<F> {
    pub alpha: F,
    pub momentum: F,
    pub momentum_sgd_namespace_id: &'static str,
}

impl<'t, 'g, F: Float> MomentumSGD<F> {
    /// Instantiates `Adam` optimizer with the recommended parameters in the original paper.
    pub fn default(
        unique_namespace_id: &'static str,
        var_id_list: impl IntoIterator<Item = VariableID>,
        env_handle: &mut VariableEnvironment<F>,
    ) -> MomentumSGD<F> {
        MomentumSGD::new(
            F::from(0.01).unwrap(),
            F::from(0.9).unwrap(),
            var_id_list,
            env_handle,
            unique_namespace_id,
        )
    }

    /// Instantiates `MomentumSGD` optimizer with given params.
    pub fn new(
        alpha: F,
        momentum: F,
        var_id_list: impl IntoIterator<Item = VariableID>,
        env: &mut VariableEnvironment<F>,
        momentum_sgd_namespace_id: &'static str,
    ) -> MomentumSGD<F> {
        for vid in var_id_list.into_iter() {
            let v_name = format!("{}", vid);
            let v = {
                let target_var = env
                    .get_array_by_id(vid)
                    .expect("variable array not found")
                    .borrow();
                let var_shape = target_var.shape();
                crate::ndarray_ext::zeros(var_shape)
            };
            let mut ns = env.namespace_mut(momentum_sgd_namespace_id);
            ns.slot().name(v_name).set(v);
        }
        MomentumSGD {
            alpha,
            momentum,
            momentum_sgd_namespace_id,
        }
    }
}

impl<F: Float> Optimizer<F> for MomentumSGD<F> {
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
            let namespace = g.env().namespace(self.momentum_sgd_namespace_id);
            let var_id = param.get_variable_id().expect("Got non-variable tensor");
            let v = g.variable_by_name(&format!("{}", var_id), &namespace);

            ret.push(
                Tensor::builder(g)
                    .append_input(param, true)
                    .append_input(grads[i].as_ref(), false)
                    .append_input(&v, true)
                    .build(sgd::MomentumSGDOp {
                        lr: self.alpha,
                        momentum: self.momentum,
                    }),
            );
        }
        ret
    }
}
