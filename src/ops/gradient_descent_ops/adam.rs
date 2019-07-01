//! Module defining Adam optimizer
extern crate ndarray;

use crate::ndarray_ext::NdArray;
use crate::tensor::Tensor;
use crate::Float;
use std::cell::Cell;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;

struct AdamOp<T: Float> {
    static_params: StaticParams<T>,
    // `t` param in the original paper
    t: Cell<T>,
}

impl<T: Float> crate::op::Op<T> for AdamOp<T> {
    fn name(&self) -> &str {
        "Adam"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> crate::op::ComputeResults<'v, T> {
        let StaticParams { alpha, eps, b1, b2 } = self.static_params;
        let xs = ctx.grab_inputs();
        let t = self.t.get();

        // Make new m
        let new_m = {
            let tmp = T::one() - b1;
            unsafe {
                if let Some(arr) = ctx.node(2).get_persistent_array_mut() {
                    arr.zip_mut_with(&xs[1], move |x2_elem, &g| {
                        *x2_elem = *x2_elem * b1 + tmp * g
                    });
                }
            }
            // m is not empty
            ctx.node(2).get_persistent_array().unwrap()
        };

        // Make new v
        let new_v = {
            let tmp = T::one() - b2;
            unsafe {
                if let Some(arr) = ctx.node(3).get_persistent_array_mut() {
                    arr.zip_mut_with(&xs[1], move |x3_elem, &g| {
                        *x3_elem = *x3_elem * b2 + tmp * g * g
                    });
                }
            }
            // v is not empty
            ctx.node(3).get_persistent_array().unwrap()
        };

        // Make hat
        let m_hat = {
            let rhs = T::one() / (T::one() - b2.powf(t));
            let v_hat = new_v.mapv(move |new_v_elem| new_v_elem * rhs);
            let rhs = T::one() / (T::one() - b1.powf(t));
            let mut m_hat = new_m.mapv(move |new_m_elem| new_m_elem * rhs);
            m_hat.zip_mut_with(&v_hat, move |a, &b| (*a) /= b.sqrt() + eps);
            m_hat
        };

        // Update t and variable
        unsafe {
            if let Some(arr) = ctx.node(0).get_persistent_array_mut() {
                arr.zip_mut_with(&m_hat, move |l, &r| *l -= alpha * r);
            }
        }
        self.t.set(t + T::one());

        vec![Err(crate::op::ComputeException::NoOutput)]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

/// Use `Adam::vars_with_states` to instantiate this.
#[doc(hidden)]
pub struct StatefulVariable<'a, T: Float + 'a> {
    pub var: &'a Tensor<T>,
    pub state: StatefulParams<T>,
}

/// Adam optimizer
///
/// The implementation is based on http://arxiv.org/abs/1412.6980v8.
/// Please create state objects by use of `Adam::vars_with_states` beforehand.
///
/// ```
/// extern crate autograd as ag;
///
/// // Define parameters to optimize.
/// let w: ag::Tensor<f32> = ag::variable(ag::ndarray_ext::glorot_uniform(&[28 * 28, 10]));
/// let b: ag::Tensor<f32> = ag::variable(ag::ndarray_ext::zeros(&[1, 10]));
/// let params = ag::gradient_descent_ops::Adam::vars_with_states(&[&w, &b]);
///
/// // Create update ops.
/// let adam = ag::gradient_descent_ops::Adam::<f32>::default();
/// // let update_ops: &[Tensor<f32>] = &adam.compute_updates(&params, grads);
/// ```
///
/// See also https://github.com/raskr/rust-autograd/blob/master/examples/mlp_mnist.rs
pub struct Adam<T: Float> {
    pub alpha: T,
    pub eps: T,
    pub b1: T,
    pub b2: T,
}

impl<T: Float> Default for Adam<T> {
    /// Instantiates `Adam` optimizer with the recommended parameters in the original paper.
    fn default() -> Adam<T> {
        Adam {
            alpha: T::from(0.001).unwrap(),
            eps: T::from(1e-08).unwrap(),
            b1: T::from(0.9).unwrap(),
            b2: T::from(0.999).unwrap(),
        }
    }
}

impl<T: Float> Adam<T> {
    /// Creates stateful variable tensors used for Adam optimizer.
    pub fn vars_with_states<'a>(tensors: &[&'a Tensor<T>]) -> Vec<StatefulVariable<'a, T>> {
        let mut var2state = BTreeMap::<super::StateKey<'a, T>, StatefulParams<T>>::new();
        tensors
            .into_iter()
            .map(|var| {
                // let var = var.as_ref();
                if let Some(var_arr) = var.get_persistent_array() {
                    match var2state.entry(super::StateKey(var)) {
                        Entry::Vacant(ent) => {
                            let inserted = ent.insert(StatefulParams {
                                m: crate::ops::variable(NdArray::zeros(var_arr.shape())),
                                v: crate::ops::variable(NdArray::zeros(var_arr.shape())),
                            });
                            StatefulVariable {
                                var,
                                state: inserted.clone(),
                            }
                        }
                        Entry::Occupied(ent) => StatefulVariable {
                            var,
                            state: ent.get().clone(),
                        },
                    }
                } else {
                    panic!("Can't optimize non-variable.")
                }
            })
            .collect()
    }

    /// Creates ops to optimize `params` with Adam.
    ///
    /// Evaluated results of the return values will be `None`.
    pub fn compute_updates<A: AsRef<Tensor<T>>>(
        &self,
        params: &[StatefulVariable<T>],
        grads: &[A],
    ) -> Vec<Tensor<T>> {
        params
            .into_iter()
            .zip(grads)
            .map(|(param, grad)| {
                let StatefulParams { ref m, ref v } = param.state;
                Tensor::builder()
                    .set_inputs(vec![param.var, grad.as_ref(), m, v])
                    .build(AdamOp {
                        t: Cell::new(T::one()),
                        static_params: StaticParams {
                            alpha: self.alpha,
                            eps: self.eps,
                            b1: self.b1,
                            b2: self.b2,
                        },
                    })
            })
            .collect()
    }
}

/// Holds Adam's static parameters (`alpha`, `eps`, `b1`, `b2`)
#[derive(Copy, Clone)]
#[doc(hidden)]
pub struct StaticParams<T: Float> {
    pub alpha: T,
    pub eps: T,
    pub b1: T,
    pub b2: T,
}

/// Wrapper of state objects in Adam's computation (`m` and `v`)
#[derive(Clone)]
#[doc(hidden)]
pub struct StatefulParams<T: Float> {
    pub m: Tensor<T>,
    pub v: Tensor<T>,
}
