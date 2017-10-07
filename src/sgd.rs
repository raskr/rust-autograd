extern crate ndarray;
extern crate fnv;

use graph::Graph;
use ndarray_ext;
use ndarray_ext::NdArray;
use std::mem;
use tensor;
use tensor::Tensor;


#[inline]
/// Updates params with gradients
pub fn apply_gradients<T: optimizers::Optimizer>(
    variables: &[&Tensor],
    gradients: &[Tensor],
    optimizer: &mut T,
    graph: &mut Graph,
)
{
    assert!(variables.len() == gradients.len());
    let mut memo =
        mem::replace(&mut graph.memo, None).expect("Don't touch \"Graph.memo\" property");
    // run graph and get gradient arrays
    let mut grad_arrays = tensor::eval_tensors(gradients, &mut graph.variables, &mut memo);
    memo.clear();
    mem::swap(&mut Some(memo), &mut graph.memo);
    for v in variables {
        // safe unwrap
        assert_eq!(v.op.name(), "Variable", "Can't optimize non-variable");
        let mut v_arr = graph.variables.get_mut(v).unwrap();
        let g = maybe_reduce_grad(grad_arrays.remove(0), v_arr.shape());
        optimizer.update(v, v_arr, g);
    }
}


#[doc(hidden)]
#[inline(always)]
/// Reduces gradient's each dim by summation.
/// This is used when parameter shape and
/// gradient shape are not same due to broadcast.
pub fn maybe_reduce_grad(mut grad: NdArray, var_shape: &[usize]) -> NdArray
{
    let grad_shape = grad.shape().to_vec();
    // for each grad axis
    for (i, (g, v)) in grad_shape.iter().zip(var_shape).enumerate() {
        if g == v {
            continue; // do nothing
        } else if g < v {
            panic!("bad gradient")
        } else {
            grad = ndarray_ext::expand_dims(grad.sum(ndarray::Axis(i)), i);
        }
    }
    grad
}


pub mod optimizers {
    use ndarray_ext::NdArray;
    use std::collections::hash_map::HashMap;
    use tensor::Tensor;

    /// Trait for any stochastic gradient descent optimizer
    pub trait Optimizer {
        #[inline]
        /// Updates the variable tensor
        ///
        /// Updates `var` with `grad`
        fn update(&mut self, node: &Tensor, param: &mut NdArray, grad: NdArray);
    }


    /// Vanilla SGD optimizer
    pub struct SGD {
        pub lr: f32,
    }

    impl Optimizer for SGD {
        #[inline]
        fn update(&mut self, _: &Tensor, param: &mut NdArray, grad: NdArray)
        {
            param.scaled_add(-self.lr, &grad);
        }
    }


    /// Adam optimizer
    ///
    /// This implementation is based on http://arxiv.org/abs/1412.6980v8
    pub struct Adam {
        // static params
        pub alpha: f32,
        pub eps: f32,
        pub b1: f32,
        pub b2: f32,
        // dynamic params
        pub states: HashMap<Tensor, AdamState>,
    }

    pub struct AdamState {
        m: NdArray,
        v: NdArray,
        t: f32,
    }

    impl Default for Adam {
        fn default() -> Adam
        {
            Adam {
                alpha: 0.001,
                eps: 1e-08,
                b1: 0.9,
                b2: 0.999,
                states: HashMap::new(),
            }
        }
    }

    impl Optimizer for Adam {
        #[inline]
        fn update(&mut self, node: &Tensor, param: &mut NdArray, mut grad: NdArray)
        {
            // get current state
            let AdamState { mut m, mut v, t } = self.states.remove(&node).unwrap_or_else(|| {
                AdamState {
                    m: NdArray::zeros(param.shape()),
                    v: NdArray::zeros(param.shape()),
                    t: 1.,
                }
            });

            // make new m
            m *= self.b1;
            m += &((1. - self.b1) * &grad);
            let m_new = m;

            // make new v
            let tmp = 1. - self.b2;
            grad.mapv_inplace(move |a| tmp * a * a);
            v *= self.b2;
            v += &grad;
            let v_new = v;

            // make hat
            let mut m_hat = &m_new / (1. - self.b1.powf(t));
            let mut v_hat = &v_new / (1. - self.b2.powf(t));

            // update states
            self.states.insert(
                node.clone(),
                AdamState {
                    m: m_new,
                    v: v_new,
                    t: t + 1.,
                },
            );

            let eps = self.eps;
            v_hat.mapv_inplace(move |a| a.sqrt() + eps);
            m_hat /= &v_hat;
            param.scaled_add(-self.alpha, &m_hat);
        }
    } // Adam end
}
