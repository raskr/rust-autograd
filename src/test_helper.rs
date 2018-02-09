extern crate ndarray;

use ndarray_ext::NdArray;
use tensor::Tensor;


/// Checks the validity of `gradients` with finite difference trick.
/// For this test only, `variables` must be "shared" variables.
pub fn gradient_check<'a, 'b, T>(
    objective: &Tensor,
    gradients: &[T],
    variables: &[&Tensor],
    feeds: &[(&Tensor, &NdArray)],
    eps: f32,
    tol: f32,
) where
    T: AsRef<Tensor>,
{
    // backprop
    let theoretical_grads = ::runtime::eval(gradients, feeds);

    // for each variable nodes
    for (var_node, th_grad) in variables.iter().zip(theoretical_grads) {
        // Sorry for unwrap
        let v_arr = unsafe { var_node.get_persistent_array_mut().unwrap() };

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
            let ref obj_pos = ::runtime::eval(&[objective], feeds)[0];

            // perturbation (-)
            unsafe {
                *head_ptr.offset(i) = evacuated - eps;
            }

            // eval
            let ref obj_neg = ::runtime::eval(&[objective], feeds)[0];

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
