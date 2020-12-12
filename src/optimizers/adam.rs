//! Adam optimizer
use crate::ops::gradient_descent_ops::adam;
use crate::tensor::{Input, Tensor, Variable};
use crate::Float;
use crate::Graph;
use crate::NdArray;
use std::sync::{Arc, RwLock};

/// Adam optimizer
///
/// This implementation is based on http://arxiv.org/abs/1412.6980v8.
///
/// ```
/// extern crate autograd as ag;
/// use ag::tensor::Variable;
/// use ag::ndarray_ext::into_shared;
/// use std::sync::{Arc, RwLock};
/// use ag::optimizers::adam;
///
/// // Define parameters to optimize.
/// let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
/// let w: Arc<RwLock<ag::NdArray<f32>>> = into_shared(rng.glorot_uniform(&[28 * 28, 10]));
/// let b: Arc<RwLock<ag::NdArray<f32>>> = into_shared(ag::ndarray_ext::zeros(&[1, 10]));
/// // Make a state of adam.
/// let state = adam::AdamState::new(&[&w, &b]);
///
/// ag::with(|g| {
///     let w_ = g.variable(w.clone());
///     let b_ = g.variable(b.clone());
///
///     // some operations using w_ and b_
///     // let y = ...
///     // let grads = ag::grad(&[y], &[w_, b_]);
///
///     // instantiate an adam optimizer with default setting.
///     let adam = adam::Adam::<f32>::default();
///
///     // Getting update ops of `params` using its gradients and adam.
///     // let update_ops: &[ag::Tensor<f32>] = &adam.compute_updates(&params, &grads, &state, &g);
/// });
/// ```
///
/// See also https://github.com/raskr/rust-autograd/blob/master/examples/
pub struct Adam<F: Float> {
    static_params: StaticParams<F>,
}

impl<T: Float> Default for Adam<T> {
    /// Instantiates `Adam` optimizer with the recommended parameters in the original paper.
    fn default() -> Adam<T> {
        let static_params = StaticParams {
            alpha: T::from(0.001).unwrap(),
            eps: T::from(1e-08).unwrap(),
            b1: T::from(0.9).unwrap(),
            b2: T::from(0.999).unwrap(),
        };
        Adam { static_params }
    }
}

impl<'t, 's: 't, F: Float> Adam<F> {
    /// Instantiates Adam from static params
    pub fn new(static_params: StaticParams<F>) -> Self {
        Adam { static_params }
    }

    /// Creates ops to optimize `params` with Adam.
    ///
    /// Evaluated results of the return values will be `None`.
    pub fn compute_updates(
        &self,
        params: &[Tensor<'s, F>],
        grads: &[Tensor<'s, F>],
        states: &AdamState<F>,
        g: &'s Graph<F>,
    ) -> Vec<Tensor<'s, F>> {
        let num_params = params.len();
        let mut ret = Vec::with_capacity(num_params);
        for i in 0..num_params {
            let param = &params[i];
            let a: *const RwLock<NdArray<F>> = param
                .get_variable_array_ptr()
                .expect("Adam requires *variables* as its inputs.");
            let key = a as usize;
            let state = states
                .var2state
                .get(&key)
                .expect("Adam: state object wasn't fed correctly");
            let m = g.variable(state.m.clone());
            let v = g.variable(state.v.clone());
            let t = g.variable(state.t.clone());

            ret.push(
                Tensor::builder()
                    .set_inputs(&[
                        Input::new_mut(param),
                        Input::new(&grads[i]),
                        Input::new_mut(&m),
                        Input::new_mut(&v),
                        Input::new_mut(&t),
                    ])
                    .build(
                        g,
                        adam::AdamOp {
                            static_params: self.static_params.clone(),
                        },
                    ),
            );
        }
        ret
    }
}

struct StateArrays<F: Float> {
    m: Arc<RwLock<NdArray<F>>>,
    v: Arc<RwLock<NdArray<F>>>,
    t: Arc<RwLock<NdArray<F>>>,
}

/// A state object for an adam optimizer.
pub struct AdamState<F: Float> {
    // map key is the address of a variable array on the heap
    var2state: crate::FxHashMap<usize, StateArrays<F>>,
}

impl<F: Float> AdamState<F> {
    /// Creates a new state object for an adam optimizer.
    ///
    /// `variables` should be variable arrays fed to `autograd::variable`.
    pub fn new(variables: &[&Arc<RwLock<NdArray<F>>>]) -> Self {
        let mut map = crate::FxHashMap::default();
        for &var in variables {
            // Use the address on the heap as a hash key
            let key = ((&**var) as *const RwLock<_>) as usize;
            let var = var.read().unwrap();
            let var_shape = var.shape();
            map.insert(
                key,
                StateArrays {
                    m: Arc::new(RwLock::new(crate::ndarray_ext::zeros(var_shape))),
                    v: Arc::new(RwLock::new(crate::ndarray_ext::zeros(var_shape))),
                    t: Arc::new(RwLock::new(crate::ndarray_ext::from_scalar(F::one()))),
                },
            );
        }
        Self { var2state: map }
    }
}

/// Holds Adam's static parameters (`alpha`, `eps`, `b1`, `b2`).
#[derive(Clone)]
pub struct StaticParams<T: Float> {
    pub alpha: T,
    pub eps: T,
    pub b1: T,
    pub b2: T,
}
