extern crate ndarray;
extern crate rand;

use sgd;
use tensor;
use tensor::{Feed, Tensor};


/// This computes partial derivatives of `objective` with `var_node` using
/// back propagation, and then checks those are close to numerically
/// computed gradients (uses finite difference trick).
pub fn gradient_check(
    objective: &Tensor,
    variables: &[&Tensor],
    gradients: &[Tensor],
    feed_dict: &Feed,
    eps: f32,
    tol: f32,
)
{
    assert_eq!(variables.len(), gradients.len());

    let theoretical_grads = tensor::eval_tensors(gradients, feed_dict.clone());

    // for each variable nodes
    for (variable, theoretical_grad) in variables.iter().zip(theoretical_grads) {

        // reduce gradient if necessary
        let theoretical_grad = sgd::maybe_reduce_grad(theoretical_grad, variable);

        let var_size = variable
            .borrow()
            .param
            .as_ref()
            .expect(&format!(
                "{} is not shared variable",
                variable.borrow().op.name()
            ))
            .len();

        // for each values
        for i in 0..var_size as isize {
            let evacuated;

            let head_ptr: *mut f32 = variable.borrow_mut().param.as_mut().unwrap().as_mut_ptr();

            unsafe {
                // perturbation (+)
                evacuated = *head_ptr.offset(i);
                *head_ptr.offset(i) = evacuated + eps;
            }

            let obj_pos = objective.eval_with_input(feed_dict.clone());

            unsafe {
                // perturbation (-)
                *head_ptr.offset(i) = evacuated - eps;
            }

            let obj_neg = objective.eval_with_input(feed_dict.clone());

            unsafe {
                // restore
                *head_ptr.offset(i) = evacuated;
            }

            let g_num = (obj_pos - obj_neg).scalar_sum() / (2. * eps);
            let g_th = unsafe { *theoretical_grad.as_ptr().offset(i) };

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
