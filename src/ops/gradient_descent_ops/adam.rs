extern crate ndarray;

use ndarray_ext::NdArray;
use std::cell::Cell;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use tensor::Tensor;
use Float;

struct AdamOp<T: Float> {
    static_params: StaticParams<T>,
    t: Cell<T>,
}

impl<T: Float> ::op::Op<T> for AdamOp<T> {
    fn name(&self) -> &str {
        "Adam"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> ::op::ComputeResults<T> {
        let StaticParams { alpha, eps, b1, b2 } = self.static_params;
        let xs = ctx.grab_inputs();
        let t = self.t.get();

        // Make new m
        let new_m = {
            let tmp = T::one() - b1;
            let mut new_m = xs[2].mapv(move |x2_elem| x2_elem * b1);
            new_m.zip_mut_with(&xs[1], move |a, &g| *a += tmp * g);
            new_m
        };

        // Make new v
        let new_v = {
            let mut new_v = xs[3].mapv(move |x3_elem| x3_elem * b2);
            let tmp = T::one() - b2;
            new_v.zip_mut_with(&xs[1], move |a, &g| *a += tmp * g * g);
            new_v
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

        // Update t and params
        unsafe {
            ::ndarray_ext::axpy(&xs[0], -alpha, m_hat.as_ptr(), m_hat.shape()); // variable
            ::ndarray_ext::assign(&xs[2], new_m.as_ptr(), new_m.shape());
            ::ndarray_ext::assign(&xs[3], new_v.as_ptr(), new_v.shape());
            self.t.set(t + T::one());
        }

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
}
