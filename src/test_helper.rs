extern crate ndarray;
extern crate rand;

use graph;
use ndarray_ext::NdArray;
use sgd;
use std::mem;
use tensor::Tensor;


#[allow(mutable_transmutes)]
/// This computes partial derivatives of `objective` with `var_node` using
/// back propagation, and then checks those are close to numerically
/// computed gradients (uses finite difference trick).
pub fn gradient_check(
    objective: &Tensor,
    gradients: &[Tensor],
    variables: &[&Tensor],
    graph: graph::Graph,
    eps: f32,
    tol: f32,
)
{
    // back prop
    let gradients = gradients.iter().map(|a| a).collect::<Vec<_>>();
    let theoretical_grads = graph.eval_keep_feeds(gradients.as_slice());

    // for each variable nodes
    for (v, th_grad) in variables.iter().zip(theoretical_grads) {
        let v_arr = graph.variables.get(v).expect("You passed non-variable");
        let mut v_arr = unsafe { mem::transmute::<&NdArray, &mut NdArray>(v_arr) };

        // reduce gradient if necessary
        let th_grad = sgd::maybe_reduce_grad(th_grad, v_arr.shape());

        let head_ptr: *mut f32 = v_arr.as_mut_ptr();

        // for each values
        for i in 0..v_arr.len() as isize {
            let evacuated;

            // perturbation (+)
            unsafe {
                evacuated = *head_ptr.offset(i);
                *head_ptr.offset(i) = evacuated + eps;
            }

            // eval
            let ref obj_pos = graph.eval_keep_feeds(&[objective])[0];

            // perturbation (-)
            unsafe {
                *head_ptr.offset(i) = evacuated - eps;
            }

            // eval
            let ref obj_neg = graph.eval_keep_feeds(&[objective])[0];

            // restore
            unsafe {
                *head_ptr.offset(i) = evacuated;
            }

            let g_num = (obj_pos - obj_neg).scalar_sum() / (2. * eps);
            let g_th = unsafe { *th_grad.as_ptr().offset(i) };

            // compare
            let diff = (g_num - g_th).abs();
            if diff > tol {
                panic!(
                    "Gradient checking failed with too large error: num={}, bp={}",
                    g_num,
                    g_th
                );
            }
        }
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! eval_with_time {
  ( $x:expr) => {
    {
      use std::time::{Duration, Instant};
      let start = Instant::now();
      let result = $x;
      let end = start.elapsed();
      println!("{}.{:03} sec", end.as_secs(), end.subsec_nanos() / 1_000_000);
      result
    }
  };
}
