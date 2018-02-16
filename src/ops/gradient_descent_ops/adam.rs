extern crate ndarray;

use ndarray_ext::NdArray;
use op;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use tensor::Tensor;

struct AdamOp
{
    static_params: StaticParams,
}

impl ::op::Op for AdamOp
{
    fn name(&self) -> &str
    {
        "Adam"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let StaticParams { alpha, eps, b1, b2 } = self.static_params;
        let mut xs = unsafe { ctx.grab_assignable_inputs() };

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
        vec![Err(::errors::OpComputeErrorStatus::NoOutput)]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

/// Adam optimizer
///
/// This implementation is based on http://arxiv.org/abs/1412.6980v8
pub struct Adam<'a>
{
    pub alpha:           f32,
    pub eps:             f32,
    pub b1:              f32,
    pub b2:              f32,
    pub stateful_params: BTreeMap<super::StateKey<'a>, StatefulParams>,
}

impl<'a> Default for Adam<'a>
{
    fn default() -> Adam<'a>
    {
        Adam {
            alpha:           0.001,
            eps:             1e-08,
            b1:              0.9,
            b2:              0.999,
            stateful_params: BTreeMap::new(),
        }
    }
}

#[derive(Copy, Clone)]
pub struct StaticParams
{
    pub alpha: f32,
    pub eps:   f32,
    pub b1:    f32,
    pub b2:    f32,
}

pub struct StatefulParams
{
    pub m: Tensor,
    pub v: Tensor,
    pub t: Tensor, // shape: []
}

impl<'a> super::Optimizer<'a> for Adam<'a>
{
    fn compute_updates<T: AsRef<Tensor>>(
        &mut self,
        params: &[&'a Tensor],
        grads: &[T],
    ) -> Vec<Tensor>
    {
        params
            .into_iter()
            .zip(grads)
            .map(|(param, grad)| {
                let op = AdamOp {
                    static_params: StaticParams {
                        alpha: self.alpha,
                        eps:   self.eps,
                        b1:    self.b1,
                        b2:    self.b2,
                    },
                };

                if let Some(ref param_arr) = param.persistent_array {
                    match self.stateful_params.entry(super::StateKey(param)) {
                        Entry::Vacant(ent) => {
                            let StatefulParams {
                                ref m,
                                ref v,
                                ref t,
                            } = *ent.insert(StatefulParams {
                                m: ::ops::variable(NdArray::zeros(param_arr.shape())),
                                v: ::ops::variable(NdArray::zeros(param_arr.shape())),
                                t: ::ops::variable(::ndarray_ext::from_scalar(1.)),
                            });
                            Tensor::builder()
                                .set_inputs(vec![param, grad.as_ref(), m, v, t])
                                .build(op)
                        }
                        Entry::Occupied(ent) => {
                            let StatefulParams {
                                ref m,
                                ref v,
                                ref t,
                            } = *ent.get();
                            Tensor::builder()
                                .set_inputs(vec![param, grad.as_ref(), m, v, t])
                                .build(op)
                        }
                    }
                } else {
                    panic!("Can't optimize non-variable.")
                }
            })
            .collect()
    }
}
