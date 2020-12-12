//! A small extension of [ndarray](https://github.com/rust-ndarray/ndarray)
//!
//! Mainly provides `array_gen`, which is a collection of array generator functions.
use crate::Float;
use ndarray;
use std::sync::{Arc, RwLock};

/// alias for `ndarray::Array<T, IxDyn>`
pub type NdArray<T> = ndarray::Array<T, ndarray::IxDyn>;

/// alias for `ndarray::ArrayView<T, IxDyn>`
pub type NdArrayView<'a, T> = ndarray::ArrayView<'a, T, ndarray::IxDyn>;

/// alias for `ndarray::ArrayViewMut<T, IxDyn>`
pub type NdArrayViewMut<'a, T> = ndarray::ArrayViewMut<'a, T, ndarray::IxDyn>;

// expose array_gen
pub use crate::array_gen::*;

/// `Op::compute`'s output
#[derive(Clone)]
pub(crate) enum ArrRepr<'v, T: Float> {
    /// Represents `ndarray::Array<T: Float, ndarray::IxDyn>`
    Owned(NdArray<T>),

    /// Represents `ndarray::ArrayView<'a, T: Float, ndarray::IxDyn>`
    View(NdArrayView<'v, T>),
}

/// Helper to feed ndarrays into variable tensors.
///
/// Converts the input array into `ndarray::Array<F, ndarray::IxDyn>`
/// and then simply wraps it with `Arc<RwLock<...>>`.
///
/// ```
/// use autograd as ag;
/// use ag::ndarray_ext as array;
/// use ag::tensor::Variable;
/// use ndarray;
/// use std::sync::{Arc, RwLock};
///
/// type NdArray = ndarray::Array<f32, ndarray::IxDyn>;
///
/// let w_arr: Arc<RwLock<NdArray>> = array::into_shared(array::ones(&[28 * 28, 10]));
/// ag::with(|g| {
///     let w = g.variable(w_arr.clone());
/// });
///
/// ```
#[inline]
pub fn into_shared<F: Float, D: ndarray::Dimension>(
    arr: ndarray::Array<F, D>,
) -> Arc<RwLock<NdArray<F>>> {
    Arc::new(RwLock::new(arr.into_dyn()))
}

#[inline]
/// This works well only for small arrays
pub(crate) fn as_shape<T: Float>(x: &NdArrayView<T>) -> Vec<usize> {
    let mut target = Vec::with_capacity(x.len());
    for &a in x.iter() {
        target.push(a.to_usize().unwrap());
    }
    target
}

#[inline]
pub(crate) fn expand_dims_view<T: Float>(x: NdArrayView<T>, axis: usize) -> NdArrayView<T> {
    let mut shape = x.shape().to_vec();
    shape.insert(axis, 1);
    x.into_shape(shape).unwrap()
}

#[inline]
pub(crate) fn expand_dims<T: Float>(x: NdArray<T>, axis: usize) -> NdArray<T> {
    let mut shape = x.shape().to_vec();
    shape.insert(axis, 1);
    x.into_shape(shape).unwrap()
}

#[inline]
pub(crate) fn roll_axis<T: Float>(arg: &mut NdArray<T>, to: ndarray::Axis, from: ndarray::Axis) {
    let i = to.index();
    let mut j = from.index();
    if j > i {
        while i != j {
            arg.swap_axes(i, j);
            j -= 1;
        }
    } else {
        while i != j {
            arg.swap_axes(i, j);
            j += 1;
        }
    }
}

#[inline]
pub(crate) fn normalize_negative_axis(axis: isize, ndim: usize) -> usize {
    if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    }
}

#[inline]
pub(crate) fn normalize_negative_axes<T: Float>(axes: &NdArrayView<T>, ndim: usize) -> Vec<usize> {
    let mut axes_ret: Vec<usize> = Vec::with_capacity(axes.len());
    for &axis in axes.iter() {
        let axis = if axis < T::zero() {
            (T::from(ndim).unwrap() + axis)
                .to_usize()
                .expect("Invalid index value")
        } else {
            axis.to_usize().expect("Invalid index value")
        };
        axes_ret.push(axis);
    }
    axes_ret
}

#[inline]
pub(crate) fn sparse_to_dense<T: Float>(arr: &NdArrayView<T>) -> Vec<usize> {
    let mut axes: Vec<usize> = vec![];
    for (i, &a) in arr.iter().enumerate() {
        if a == T::one() {
            axes.push(i);
        }
    }
    axes
}

#[allow(unused)]
#[inline]
pub(crate) fn is_fully_transposed(strides: &[ndarray::Ixs]) -> bool {
    let mut ret = true;
    for w in strides.windows(2) {
        if w[0] > w[1] {
            ret = false;
            break;
        }
    }
    ret
}

#[inline]
pub(crate) fn copy_if_not_standard<T: Float>(x: &NdArrayView<T>) -> Option<NdArray<T>> {
    if !x.is_standard_layout() {
        Some(deep_copy(x))
    } else {
        None
    }
}

#[inline]
pub(crate) fn deep_copy<T: Float>(x: &NdArrayView<T>) -> NdArray<T> {
    let vec = x.iter().cloned().collect::<Vec<_>>();
    // tested
    unsafe { NdArray::from_shape_vec_unchecked(x.shape(), vec) }
}

#[inline]
pub(crate) fn scalar_shape<T: Float>() -> NdArray<T> {
    // tested
    unsafe { NdArray::from_shape_vec_unchecked(ndarray::IxDyn(&[0]), vec![]) }
}

#[inline]
pub(crate) fn is_scalar_shape(shape: &[usize]) -> bool {
    shape == [] || shape == [0]
}

#[inline]
pub(crate) fn shape_of_view<T: Float>(x: &NdArrayView<T>) -> NdArray<T> {
    let shape = x
        .shape()
        .iter()
        .map(|&a| T::from(a).unwrap())
        .collect::<Vec<T>>();
    let rank = shape.len();
    // safe unwrap
    NdArray::from_shape_vec(ndarray::IxDyn(&[rank]), shape).unwrap()
}

#[inline]
pub(crate) fn shape_of<T: Float>(x: &NdArray<T>) -> NdArray<T> {
    let shape = x
        .shape()
        .iter()
        .map(|&a| T::from(a).unwrap())
        .collect::<Vec<T>>();
    let rank = shape.len();
    // safe unwrap
    NdArray::from_shape_vec(ndarray::IxDyn(&[rank]), shape).unwrap()
}

#[cfg(feature = "mkl")]
#[inline]
pub(crate) fn get_batch_ptrs_mut<A: Float, B>(
    batch_size: usize,
    head: *mut A,
    whole_size: usize,
) -> Vec<*mut B> {
    let size_per_sample = whole_size / batch_size;
    let mut ret = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        unsafe {
            ret.push(head.offset((i * size_per_sample) as isize) as *mut B);
        }
    }
    ret
}

#[cfg(feature = "mkl")]
#[inline]
pub(crate) fn get_batch_ptrs<A: Float, B>(
    batch_size: usize,
    head: *const A,
    whole_size: usize,
) -> Vec<*const B> {
    let size_per_sample = whole_size / batch_size;
    let mut ret = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        unsafe {
            ret.push(head.offset((i * size_per_sample) as isize) as *const B);
        }
    }
    ret
}

/// A collection of array generator functions.
pub mod array_gen {
    use super::*;
    use rand::distributions::Distribution;
    use rand::rngs::ThreadRng;
    use rand::{self, Rng};
    use rand_distr;
    use std::marker::PhantomData;
    use std::sync::Mutex;

    /// Helper structure to create ndarrays whose elements are pseudorandom numbers.
    ///
    /// This is actually a wrapper of an arbitrary `rand::Rng`, the default is `rand::rngs::ThreadRng`.
    ///
    /// ```
    /// use autograd as ag;
    /// use rand;
    ///
    /// type NdArray = ndarray::Array<f32, ndarray::IxDyn>;
    ///
    /// let my_rng = ag::ndarray_ext::ArrayRng::new(rand::thread_rng());
    /// let random: NdArray = my_rng.standard_normal(&[2, 3]);
    ///
    /// // The default is `ThreadRng` (seed number is not fixed).
    /// let default = ag::ndarray_ext::ArrayRng::default();
    /// let random: NdArray = default.standard_normal(&[2, 3]);
    /// ```
    pub struct ArrayRng<T: Float, R: Rng = ThreadRng> {
        phantom: PhantomData<T>,
        rng: Mutex<R>,
    }

    impl<T: Float> Default for ArrayRng<T> {
        /// Initialize with `rand::rngs::ThreadRng`.
        fn default() -> Self {
            ArrayRng {
                phantom: PhantomData,
                rng: Mutex::new(rand::thread_rng()),
            }
        }
    }

    impl<T: Float, R: Rng> ArrayRng<T, R> {
        /// Creates `ArrRng` with pre-instantiated `Rng`.
        pub fn new(rng: R) -> Self {
            ArrayRng {
                phantom: PhantomData,
                rng: Mutex::new(rng),
            }
        }

        /// Generates `ndarray::Array<T, ndarray::IxDyn>` whose elements are random numbers.
        fn gen_random_array<I>(&self, shape: &[usize], dist: I) -> NdArray<T>
        where
            I: Distribution<f64>,
        {
            let size: usize = shape.iter().cloned().product();
            let mut rng = self.rng.lock().unwrap();
            unsafe {
                let mut buf = Vec::with_capacity(size);
                for i in 0..size {
                    *buf.get_unchecked_mut(i) = T::from(dist.sample(&mut *rng)).unwrap();
                }
                buf.set_len(size);
                NdArray::from_shape_vec_unchecked(shape, buf)
            }
        }

        /// Creates an ndarray sampled from the normal distribution with given params.
        pub fn random_normal(
            &self,
            shape: &[usize],
            mean: f64,
            stddev: f64,
        ) -> ndarray::Array<T, ndarray::IxDyn> {
            let normal = rand_distr::Normal::new(mean, stddev).unwrap();
            self.gen_random_array(shape, normal)
        }

        /// Creates an ndarray sampled from the uniform distribution with given params.
        pub fn random_uniform(
            &self,
            shape: &[usize],
            min: f64,
            max: f64,
        ) -> ndarray::Array<T, ndarray::IxDyn> {
            let range = rand_distr::Uniform::new(min, max);
            self.gen_random_array(shape, range)
        }

        /// Creates an ndarray sampled from the standard normal distribution.
        pub fn standard_normal(&self, shape: &[usize]) -> ndarray::Array<T, ndarray::IxDyn> {
            let normal = rand_distr::Normal::new(0., 1.).unwrap();
            self.gen_random_array(shape, normal)
        }

        /// Creates an ndarray sampled from the standard uniform distribution.
        pub fn standard_uniform(&self, shape: &[usize]) -> ndarray::Array<T, ndarray::IxDyn> {
            let dist = rand_distr::Uniform::new(0., 1.);
            self.gen_random_array(shape, dist)
        }

        /// Glorot normal initialization. (a.k.a. Xavier normal initialization)
        pub fn glorot_normal(&self, shape: &[usize]) -> ndarray::Array<T, ndarray::IxDyn> {
            assert_eq!(shape.len(), 2);
            let s = 1. / (shape[0] as f64).sqrt();
            let normal = rand_distr::Normal::new(0., s).unwrap();
            self.gen_random_array(shape, normal)
        }

        /// Glorot uniform initialization. (a.k.a. Xavier uniform initialization)
        pub fn glorot_uniform(&self, shape: &[usize]) -> ndarray::Array<T, ndarray::IxDyn> {
            assert_eq!(shape.len(), 2);
            let s = (6. / shape[0] as f64).sqrt();
            let uniform = rand_distr::Uniform::new(-s, s);
            self.gen_random_array(shape, uniform)
        }

        /// Creates an ndarray sampled from the bernoulli distribution with given params.
        pub fn bernoulli(&self, shape: &[usize], p: f64) -> ndarray::Array<T, ndarray::IxDyn> {
            let dist = rand_distr::Uniform::new(0., 1.);
            let mut rng = self.rng.lock().unwrap();
            let size: usize = shape.iter().cloned().product();
            unsafe {
                let mut buf = Vec::with_capacity(size);
                for i in 0..size {
                    let val = dist.sample(&mut *rng);
                    *buf.get_unchecked_mut(i) = T::from(i32::from(val < p)).unwrap();
                }
                buf.set_len(size);
                NdArray::from_shape_vec(shape, buf).unwrap()
            }
        }

        /// Creates an ndarray sampled from the exponential distribution with given params.
        pub fn exponential(
            &self,
            shape: &[usize],
            lambda: f64,
        ) -> ndarray::Array<T, ndarray::IxDyn> {
            let dist = rand_distr::Exp::new(lambda).unwrap();
            self.gen_random_array(shape, dist)
        }

        /// Creates an ndarray sampled from the log normal distribution with given params.
        pub fn log_normal(
            &self,
            shape: &[usize],
            mean: f64,
            stddev: f64,
        ) -> ndarray::Array<T, ndarray::IxDyn> {
            let dist = rand_distr::LogNormal::new(mean, stddev).unwrap();
            self.gen_random_array(shape, dist)
        }

        /// Creates an ndarray sampled from the gamma distribution with given params.
        pub fn gamma(
            &self,
            shape: &[usize],
            shape_param: f64,
            scale: f64,
        ) -> ndarray::Array<T, ndarray::IxDyn> {
            let dist = rand_distr::Gamma::new(shape_param, scale).unwrap();
            self.gen_random_array(shape, dist)
        }
    }

    #[inline]
    /// Creates an ndarray filled with 0s.
    pub fn zeros<T: Float>(shape: &[usize]) -> ndarray::Array<T, ndarray::IxDyn> {
        NdArray::from_elem(shape, T::zero())
    }

    #[inline]
    /// Creates an ndarray filled with 1s.
    pub fn ones<T: Float>(shape: &[usize]) -> ndarray::Array<T, ndarray::IxDyn> {
        NdArray::from_elem(shape, T::one())
    }

    #[inline]
    /// Creates an ndarray object from a scalar.
    pub fn from_scalar<T: Float>(val: T) -> ndarray::Array<T, ndarray::IxDyn> {
        NdArray::from_elem(ndarray::IxDyn(&[]), val)
    }
}
