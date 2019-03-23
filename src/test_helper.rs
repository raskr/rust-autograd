use ndarray_ext::NdArray;
use std::cmp::Ordering;
use std::collections::btree_set::BTreeSet;
use tensor::Tensor;
use Float;

/// Checks the validity of `gradients` with finite difference trick.
/// For this test only, `variables` must be "shared" variables.
pub fn check_theoretical_grads<'a, 'b, A, T>(
    objective: &Tensor<T>,
    gradients: &[A],
    variables: &[&Tensor<T>],
    feeds: &[(&Tensor<T>, &NdArray<T>)],
    eps: T,
    tol: T,
) where
    A: AsRef<Tensor<T>>,
    T: Float,
{
    // backprop
    let theoretical_grads = ::runtime::eval(gradients, feeds);

    // for each variable nodes
    for (var_node, th_grad) in variables.iter().zip(theoretical_grads) {
        let mut v_arr = unsafe {
            var_node
                .get_persistent_array_mut()
                .expect("This is not a variable")
        };
        let head_ptr: *mut T = v_arr.as_mut_ptr();

        // for each values
        for i in 0..v_arr.len() as isize {
            let evacuated;

            // perturbation (+)
            unsafe {
                evacuated = *head_ptr.offset(i);
                *head_ptr.offset(i) = evacuated + eps;
            }

            // eval
            let obj_pos = objective.eval(feeds).unwrap();

            // perturbation (-)
            unsafe {
                *head_ptr.offset(i) = evacuated - eps;
            }

            // eval
            let obj_neg = objective.eval(feeds).unwrap();

            // restore
            unsafe {
                *head_ptr.offset(i) = evacuated;
            }

            let two = T::one() + T::one();
            let g_num = (obj_pos - obj_neg).scalar_sum() / (two * eps);
            let g_th = unsafe { *th_grad.as_ref().unwrap().as_ptr().offset(i) };

            // compare
            let diff = (g_num - g_th).abs();
            if diff > tol {
                panic!(
                    "Gradient checking failed with too large error: num={}, bp={}",
                    g_num, g_th
                );
            }
        }
    }
}

/// Traverse a graph from endpoint "t".
pub fn visit_once<F, T: Float>(t: &Tensor<T>, f: &mut F)
where
    F: FnMut(&Tensor<T>) -> (),
{
    visit_once_internal(t, f, &mut BTreeSet::new())
}

fn visit_once_internal<'a, F, T: Float>(
    t: &'a Tensor<T>,
    f: &mut F,
    visited: &mut BTreeSet<&'a Tensor<T>>,
) where
    F: FnMut(&'a Tensor<T>) -> (),
{
    if visited.contains(&t) {
        return; // exit early
    } else {
        visited.insert(t); // first visit
    }

    f(&t);

    for child in t.inputs.iter() {
        visit_once_internal(child, f, visited)
    }
}

impl<'a, T: Float> Ord for &'a Tensor<T> {
    #[inline]
    /// Compares the addresses of the two tensors.
    /// This can be used for ordering-based data structures (e.g. BinaryTree).
    fn cmp(&self, other: &&'a Tensor<T>) -> Ordering {
        let a = (*self) as *const Tensor<T>;
        let b = (*other) as *const Tensor<T>;
        a.cmp(&b)
    }
}

impl<'a, T: Float> PartialOrd for &'a Tensor<T> {
    #[inline]
    /// Compares the addresses of the two tensors.
    /// This can be used for ordering-based data structures (e.g. BinaryTree).
    fn partial_cmp(&self, other: &&'a Tensor<T>) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! eval_with_time {
    ($x:expr) => {{
        use std::time::{Duration, Instant};
        let start = Instant::now();
        let result = $x;
        let end = start.elapsed();
        println!(
            "{}.{:03} sec",
            end.as_secs(),
            end.subsec_nanos() / 1_000_000
        );
        result
    }};
}
