extern crate ndarray;

use ndarray_ext::NdArray;
use op;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use tensor::Tensor;
use Float;

struct AdamOp<T: Float> {
    static_params: StaticParams<T>,
}

impl<T: Float> ::op::Op<T> for AdamOp<T> {
    fn name(&self) -> &str {
        "Adam"
    }

    fn compute(&self, mut ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let StaticParams { alpha, eps, b1, b2 } = self.static_params;
        let xs = unsafe { ctx.grab_assignable_inputs() };

        // Make new m
        let new_m = {
            let mut new_m = (xs[2] as &NdArray<T>).mapv(move |x2_elem| x2_elem * b1);
            let tmp = T::one() - b1;
            new_m.zip_mut_with(xs[1], move |a, &g| *a += tmp * g);
            new_m
        };

        // Make new v
        let new_v = {
            let mut new_v = (xs[3] as &NdArray<T>).mapv(move |x3_elem| x3_elem * b2);
            let tmp = T::one() - b2;
            new_v.zip_mut_with(xs[1], move |a, &g| *a += tmp * g * g);
            new_v
        };

        // Make hat
        let m_hat = {
            let t = xs[4][ndarray::IxDyn(&[])];
            let rhs = T::one() / (T::one() - b2.powf(t));
            let v_hat = new_v.mapv(move |new_v_elem| new_v_elem * rhs);
            let rhs = T::one() / (T::one() - b1.powf(t));
            let mut m_hat = new_m.mapv(move |new_m_elem| new_m_elem * rhs);
            m_hat.zip_mut_with(&v_hat, move |a, &b| (*a) /= b.sqrt() + eps);
            m_hat
        };

        // Update t and param
        xs[4][ndarray::IxDyn(&[])] += T::one();
        xs[0].scaled_add(-alpha, &m_hat);
        vec![Err(::op::ComputeException::NoOutput)]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

pub struct StatefulVariable<'a, T: Float + 'a> {
    pub var: &'a Tensor<T>,
    pub state: StatefulParams<T>,
}

/// Adam optimizer
///
/// The implementation is based on http://arxiv.org/abs/1412.6980v8
pub struct Adam<T: Float> {
    pub alpha: T,
    pub eps: T,
    pub b1: T,
    pub b2: T,
}

impl<T: Float> Default for Adam<T> {
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
                                m: ::ops::variable(NdArray::zeros(var_arr.shape())),
                                v: ::ops::variable(NdArray::zeros(var_arr.shape())),
                                t: ::ops::variable(::ndarray_ext::from_scalar(T::one())),
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
    pub fn compute_updates<A: AsRef<Tensor<T>>>(
        &self,
        params: &[StatefulVariable<T>],
        grads: &[A],
    ) -> Vec<Tensor<T>> {
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
pub struct StaticParams<T: Float> {
    pub alpha: T,
    pub eps: T,
    pub b1: T,
    pub b2: T,
}

#[derive(Clone)]
pub struct StatefulParams<T: Float> {
    pub m: Tensor<T>,
    pub v: Tensor<T>,
    pub t: Tensor<T>,
}
