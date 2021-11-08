//! Adam optimizer
use crate::evaluation::Feeder;
use crate::optimizers::Optimizer;
use crate::tensor::Tensor;
use crate::tensor_ops::gradient_descent_ops::adam;
use crate::variable::VariableID;
use crate::{Context, Float, NdArray, VariableEnvironment};

/// Adam optimizer
///
/// This implementation is based on <http://arxiv.org/abs/1412.6980v8>.
///
/// ```
/// use autograd as ag;
///
/// use ag::prelude::*;
/// use ag::optimizers::adam;
/// use ag::variable::NamespaceTrait;
///
/// // Define parameters to optimize.
/// let mut env = ag::VariableEnvironment::new();
/// let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
///
/// let w = env.slot().set(rng.glorot_uniform(&[28 * 28, 10]));
/// let b = env.slot().set(ag::ndarray_ext::zeros(&[1, 10]));
///
/// // Adam optimizer with default params.
/// // State arrays are created in the "my_adam" namespace.
/// let adam = adam::Adam::default("my_adam", env.default_namespace().current_var_ids(), &mut env);
///
/// env.run(|g| {
///     let w = g.variable(w);
///     let b = g.variable(b);
///
///     // some operations using w and b
///     // let y = ...
///     // let grads = g.grad(&[y], &[w, b]);
///
///     // Getting update ops of `params` using its gradients and adam.
///     // let updates: &[ag::Tensor<f32>] = &adam.update(&[w, b], &grads, &g);
///
///     // for result in &g.eval(updates, &[]) {
///     //     println!("updates: {:?}", result.unwrap());
///     // }
/// });
/// ```
///
/// See also <https://github.com/raskr/rust-autograd/blob/master/examples/>
pub struct Adam<F: Float> {
    pub alpha: F,
    pub eps: F,
    pub b1: F,
    pub b2: F,
    pub adam_namespace_id: &'static str,
}

impl<'t, 'g, F: Float> Adam<F> {
    /// Instantiates `Adam` optimizer with the recommended parameters in the original paper.
    pub fn default(
        unique_namespace_id: &'static str,
        var_id_list: impl IntoIterator<Item = VariableID>,
        env_handle: &mut VariableEnvironment<F>,
    ) -> Adam<F> {
        Adam::new(
            F::from(0.001).unwrap(),
            F::from(1e-08).unwrap(),
            F::from(0.9).unwrap(),
            F::from(0.999).unwrap(),
            var_id_list,
            env_handle,
            unique_namespace_id,
        )
    }

    /// Instantiates `Adam` optimizer with given params.
    pub fn new(
        alpha: F,
        eps: F,
        b1: F,
        b2: F,
        var_id_list: impl IntoIterator<Item = VariableID>,
        env: &mut VariableEnvironment<F>,
        adam_namespace_id: &'static str,
    ) -> Adam<F> {
        for vid in var_id_list.into_iter() {
            let m_name = format!("{}m", vid);
            let v_name = format!("{}v", vid);
            let t_name = format!("{}t", vid);
            let (m, v, t) = {
                let target_var = env
                    .get_array_by_id(vid)
                    .expect("variable array not found")
                    .borrow();
                let var_shape = target_var.shape();
                (
                    crate::ndarray_ext::zeros(var_shape),
                    crate::ndarray_ext::zeros(var_shape),
                    crate::ndarray_ext::from_scalar(F::one()),
                )
            };
            let mut adam_ns = env.namespace_mut(adam_namespace_id);
            adam_ns.slot().name(m_name).set(m);
            adam_ns.slot().name(v_name).set(v);
            adam_ns.slot().name(t_name).set(t);
        }
        Adam {
            alpha,
            eps,
            b1,
            b2,
            adam_namespace_id,
        }
    }
}

impl<F: Float> Optimizer<F> for Adam<F> {
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
            let namespace = g.env().namespace(self.adam_namespace_id);
            let var_id = param.get_variable_id().expect("Got non-variable tensor");
            let m = g.variable_by_name(&format!("{}m", var_id), &namespace);
            let v = g.variable_by_name(&format!("{}v", var_id), &namespace);
            let t = g.variable_by_name(&format!("{}t", var_id), &namespace);

            ret.push(
                Tensor::builder(g)
                    .append_input(param, true)
                    .append_input(grads[i].as_ref(), false)
                    .append_input(&m, true)
                    .append_input(&v, true)
                    .append_input(&t, true)
                    .build(adam::AdamOp {
                        alpha: self.alpha,
                        eps: self.eps,
                        b1: self.b1,
                        b2: self.b2,
                    }),
            );
        }
        ret
    }
}
