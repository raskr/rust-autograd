extern crate ndarray;

use ndarray_ext::NdArray;
use op;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use tensor::Tensor;

struct AdamOp {
    static_params: StaticParams,
}

impl ::op::Op for AdamOp {
    fn name(&self) -> &str {
        "Adam"
    }

    fn compute(&self, mut ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let StaticParams { alpha, eps, b1, b2 } = self.static_params;
        let xs = unsafe { ctx.grab_assignable_inputs() };

        // Make new m
        let new_m = {
            let mut new_m = (xs[2] as &NdArray) * b1;
            let tmp = 1. - b1;
            new_m.zip_mut_with(xs[1], move |a, &g| *a += tmp * g);
            new_m
        };

        // Make new v
        let new_v = {
            let mut new_v = (xs[3] as &NdArray) * b2;
            let tmp = 1. - b2;
            new_v.zip_mut_with(xs[1], move |a, &g| *a += tmp * g * g);
            new_v
        };

        // Make hat
        let m_hat = {
            let t = xs[4][ndarray::IxDyn(&[])];
            let v_hat = new_v * (1. / (1. - b2.powf(t)));
            let mut m_hat = new_m * (1. / (1. - b1.powf(t)));
            m_hat.zip_mut_with(&v_hat, move |a, &b| (*a) /= b.sqrt() + eps);
            m_hat
        };

        // Update t and param
        xs[4][ndarray::IxDyn(&[])] += 1.;
        xs[0].scaled_add(-alpha, &m_hat);
        vec![Err(::op::ComputeException::NoOutput)]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        vec![None]
    }
}

pub struct StatefulVariable<'a> {
    pub var: &'a Tensor,
    pub state: StatefulParams,
}

/// Adam optimizer
///
/// The implementation is based on http://arxiv.org/abs/1412.6980v8
pub struct Adam {
    pub alpha: f32,
    pub eps: f32,
    pub b1: f32,
    pub b2: f32,
}

impl Default for Adam {
    fn default() -> Adam {
        Adam {
            alpha: 0.001,
            eps: 1e-08,
            b1: 0.9,
            b2: 0.999,
        }
    }
}

impl Adam {
    /// Creates stateful variable tensors used for Adam optimizer.
    pub fn vars_with_states<'a>(tensors: &[&'a Tensor]) -> Vec<StatefulVariable<'a>> {
        let mut var2state = BTreeMap::<super::StateKey<'a>, StatefulParams>::new();
        tensors
            .into_iter()
            .map(|var| {
                // let var = var.as_ref();
                if let Some(var_arr) = var.get_persistent_array() {
                    match var2state.entry(super::StateKey(var)) {
                        Entry::Vacant(ent) => {
                            let inserted = ent.insert(StatefulParams {
                                m: ::ops::variable(NdArray::zeros(var_arr.shape())),
                                v: ::ops::variable(NdArray::zeros(var_arr.shape())),
                                t: ::ops::variable(::ndarray_ext::from_scalar(1.)),
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

    // Resolves states
    pub fn compute_updates<T: AsRef<Tensor>>(
        &self,
        params: &[StatefulVariable],
        grads: &[T],
    ) -> Vec<Tensor> {
        params
            .into_iter()
            .zip(grads)
            .map(|(param, grad)| {
                let StatefulParams {
                    ref m,
                    ref v,
                    ref t,
                } = param.state;
                Tensor::builder()
                    .set_inputs(vec![param.var, grad.as_ref(), m, v, t])
                    .build(AdamOp {
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

#[derive(Copy, Clone)]
pub struct StaticParams {
    pub alpha: f32,
    pub eps: f32,
    pub b1: f32,
    pub b2: f32,
}

#[derive(Clone)]
pub struct StatefulParams {
    pub m: Tensor,
    pub v: Tensor,
    pub t: Tensor, // shape: []
}
