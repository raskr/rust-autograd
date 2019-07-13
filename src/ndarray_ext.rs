//! A small extension of rust-ndarray
//!
//! Mainly provides `array_gen`, which is a collection of array generator functions.
use crate::Float;
use ndarray;

pub type NdArray<T> = ndarray::Array<T, ndarray::IxDyn>;

pub type NdArrayView<'a, T> = ndarray::ArrayView<'a, T, ndarray::IxDyn>;

// expose array_gen
pub use crate::array_gen::*;

/// Op::compute's output
#[derive(Clone)]
pub enum ArrRepr<'v, T: Float> {
    /// Represents `ndarray::Array<T: Float, ndarray::IxDyn>`
    Owned(NdArray<T>),

    /// Represents `ndarray::ArrayView<'a, T: Float, ndarray::IxDyn>`
    View(NdArrayView<'v, T>),
}

impl<'v, T: Float> ArrRepr<'v, T> {
    #[inline]
    pub fn to_owned(&'v self) -> NdArray<T> {
        use ArrRepr::*;
        match self {
            Owned(v) => v.clone(),
            View(v) => v.to_owned(),
        }
    }

    #[inline]
    pub fn view(&'v self) -> NdArrayView<'v, T> {
        use ArrRepr::*;
        match self {
            Owned(v) => v.view(),
            View(v) => v.clone(),
        }
    }
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
            (T::from(ndim).unwrap() + axis).to_usize().unwrap()
        } else {
            axis.to_usize().unwrap()
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
// True if even one of the axes is moved
pub(crate) fn is_dims_permuted(strides: &[isize]) -> bool {
    let mut ret = false;
    for w in strides.windows(2) {
        if w[0] < w[1] {
            ret = true;
            break;
        }
    }
    ret
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
pub(crate) fn copy_if_dirty<T: Float>(x: &NdArrayView<T>) -> Option<NdArray<T>> {
    if is_dims_permuted(x.strides()) {
        Some(deep_copy(x))
    } else {
        None
    }
}

#[inline]
pub fn deep_copy<T: Float>(x: &NdArrayView<T>) -> NdArray<T> {
    let vec = x.iter().cloned().collect::<Vec<_>>();
    NdArray::from_shape_vec(x.shape(), vec).unwrap()
}

#[inline]
pub(crate) fn scalar_shape<T: Float>() -> NdArray<T> {
    // safe unwrap
    NdArray::from_shape_vec(ndarray::IxDyn(&[0]), vec![]).unwrap()
}

#[inline]
pub(crate) fn is_scalar_shape(shape: &[usize]) -> bool {
    shape == &[] || shape == &[0]
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

/// Provides a collection of array generator functions.
pub mod array_gen {
    use super::*;
    use rand::distributions::IndependentSample;
    use rand::{self, Rng, XorShiftRng};
    use std::cell::RefCell;
    use std::marker::PhantomData;


    /// Range.
    pub fn range<T: Float>(shape: &[usize]) -> NdArray<T> {
        let prod: usize = shape.iter().product();
        NdArray::<T>::from_shape_vec(
            ndarray::IxDyn(shape),
            (0..prod).map(|a| T::from(a).unwrap()).collect::<Vec<_>>(),
        ).unwrap()
    }

    /// Internal object to create ndarrays whose elements are random numbers.
    ///
    /// This is actually a wrapper of an arbitrary `rand::Rng`.
    /// You can use custom `Rng` with `new` function, whereas `default` function is provided;
    /// see https://github.com/raskr/rust-autograd/issues/1.
    pub struct ArrRng<T: Float, R = XorShiftRng> {
        phantom: PhantomData<T>,
        rng: RefCell<R>,
    }

    impl<T: Float> Default for ArrRng<T, XorShiftRng> {
        fn default() -> Self {
            ArrRng {
                phantom: PhantomData,
                rng: RefCell::new(rand::weak_rng()),
            }
        }
    }

    impl<T: Float, R: Rng> ArrRng<T, R> {
        /// Creates `ArrRng` object with `Rng` object.
        pub fn new(rng: R) -> Self {
            ArrRng {
                phantom: PhantomData,
                rng: RefCell::new(rng),
            }
        }

        #[inline]
        unsafe fn alloc(size: usize) -> Vec<T> {
            let mut buf: Vec<T> = Vec::with_capacity(size);
            buf.set_len(size);
            buf
        }

        /// Generates `ndarray::Array<T, ndarray::IxDyn>` whose elements are random numbers.
        pub fn gen_random_array<I>(&self, shape: &[usize], dist: I) -> NdArray<T>
        where
            I: IndependentSample<f64>,
        {
            let size: usize = shape.into_iter().cloned().product();
            let mut rng = self.rng.borrow_mut();
            unsafe {
                let mut buf = Self::alloc(size);
                for i in 0..size {
                    *buf.get_unchecked_mut(i) = T::from(dist.ind_sample(&mut *rng)).unwrap();
                }
                NdArray::from_shape_vec(shape, buf).unwrap()
            }
        }

        pub fn permutation(&mut self, size: usize) -> ndarray::Array1<usize> {
            let mut data: Vec<usize> = (0..size).collect();
            let slice = data.as_mut_slice();

            let mut rng = self.rng.borrow_mut();
            rng.shuffle(slice);
            ndarray::Array1::<usize>::from_vec(slice.to_vec())
        }

        pub fn random_normal(
            &self,
            shape: &[usize],
            mean: f64,
            stddev: f64,
        ) -> ndarray::Array<T, ndarray::IxDyn> {
            let normal = rand::distributions::Normal::new(mean, stddev);
            self.gen_random_array(shape, normal)
        }

        pub fn random_uniform(
            &self,
            shape: &[usize],
            min: f64,
            max: f64,
        ) -> ndarray::Array<T, ndarray::IxDyn> {
            let range = rand::distributions::Range::new(min, max);
            self.gen_random_array(shape, range)
        }

        pub fn standard_normal(&self, shape: &[usize]) -> ndarray::Array<T, ndarray::IxDyn> {
            let normal = rand::distributions::Normal::new(0., 1.);
            self.gen_random_array(shape, normal)
        }

        pub fn standard_uniform(&self, shape: &[usize]) -> ndarray::Array<T, ndarray::IxDyn> {
            let dist = rand::distributions::Range::new(0., 1.);
            self.gen_random_array(shape, dist)
        }

        pub fn glorot_normal(&self, shape: &[usize]) -> ndarray::Array<T, ndarray::IxDyn> {
            assert_eq!(shape.len(), 2);
            let s = 1. / (shape[0] as f64).sqrt();
            let normal = rand::distributions::Normal::new(0., s);
            self.gen_random_array(shape, normal)
        }

        pub fn glorot_uniform(&self, shape: &[usize]) -> ndarray::Array<T, ndarray::IxDyn> {
            assert_eq!(shape.len(), 2);
            let s = (6. / shape[0] as f64).sqrt();
            let uniform = rand::distributions::Range::new(-s, s);
            self.gen_random_array(shape, uniform)
        }

        pub fn bernoulli(&self, shape: &[usize], p: f64) -> ndarray::Array<T, ndarray::IxDyn> {
            let dist = rand::distributions::Range::new(0., 1.);
            let mut rng = self.rng.borrow_mut();
            let size: usize = shape.into_iter().cloned().product();
            unsafe {
                let mut buf = Self::alloc(size);
                for i in 0..size {
                    let val = dist.ind_sample(&mut *rng);
                    *buf.get_unchecked_mut(i) = T::from(i32::from(val < p)).unwrap();
                }
                NdArray::from_shape_vec(shape, buf).unwrap()
            }
        }

        pub fn exponential(
            &self,
            shape: &[usize],
            lambda: f64,
        ) -> ndarray::Array<T, ndarray::IxDyn> {
            let dist = rand::distributions::Exp::new(lambda);
            self.gen_random_array(shape, dist)
        }

        pub fn log_normal(
            &self,
            shape: &[usize],
            mean: f64,
            stddev: f64,
        ) -> ndarray::Array<T, ndarray::IxDyn> {
            let dist = rand::distributions::LogNormal::new(mean, stddev);
            self.gen_random_array(shape, dist)
        }

        pub fn gamma(
            &self,
            shape: &[usize],
            shape_param: f64,
            scale: f64,
        ) -> ndarray::Array<T, ndarray::IxDyn> {
            let dist = rand::distributions::Gamma::new(shape_param, scale);
            self.gen_random_array(shape, dist)
        }
    }

    #[inline]
    /// Zeros.
    pub fn zeros<T: Float>(shape: &[usize]) -> ndarray::Array<T, ndarray::IxDyn> {
        NdArray::from_elem(shape, T::zero())
    }

    #[inline]
    /// Ones.
    pub fn ones<T: Float>(shape: &[usize]) -> ndarray::Array<T, ndarray::IxDyn> {
        NdArray::from_elem(shape, T::one())
    }

    #[inline]
    /// Creates an ndarray object from a scalar.
    pub fn from_scalar<T: Float>(val: T) -> ndarray::Array<T, ndarray::IxDyn> {
        NdArray::from_elem(ndarray::IxDyn(&[]), val)
    }

    #[inline]
    /// Permutation.
    pub fn permutation(size: usize) -> ndarray::Array1<usize> {
        ArrRng::<f64>::default().permutation(size)
    }

    #[inline]
    /// Creates an ndarray sampled from a normal distribution with given params.
    pub fn random_normal<T: Float>(
        shape: &[usize],
        mean: f64,
        stddev: f64,
    ) -> ndarray::Array<T, ndarray::IxDyn> {
        ArrRng::default().random_normal(shape, mean, stddev)
    }

    #[inline]
    /// Creates an ndarray sampled from a uniform distribution with given params.
    pub fn random_uniform<T: Float>(
        shape: &[usize],
        min: f64,
        max: f64,
    ) -> ndarray::Array<T, ndarray::IxDyn> {
        ArrRng::default().random_uniform(shape, min, max)
    }

    #[inline]
    /// Creates an ndarray sampled from the standard normal distribution.
    pub fn standard_normal<T: Float>(shape: &[usize]) -> ndarray::Array<T, ndarray::IxDyn> {
        ArrRng::default().standard_normal(shape)
    }

    #[inline]
    /// Creates an ndarray sampled from the standard uniform distribution.
    pub fn standard_uniform<T: Float>(shape: &[usize]) -> ndarray::Array<T, ndarray::IxDyn> {
        ArrRng::default().standard_uniform(shape)
    }

    #[inline]
    /// Glorot normal initialization. (a.k.a. Xavier normal initialization)
    pub fn glorot_normal<T: Float>(shape: &[usize]) -> ndarray::Array<T, ndarray::IxDyn> {
        ArrRng::default().glorot_normal(shape)
    }

    #[inline]
    /// Glorot uniform initialization. (a.k.a. Xavier uniform initialization)
    pub fn glorot_uniform<T: Float>(shape: &[usize]) -> ndarray::Array<T, ndarray::IxDyn> {
        ArrRng::default().glorot_normal(shape)
    }

    /// Creates an ndarray sampled from a bernoulli distribution with given params.
    #[inline]
    pub fn bernoulli<T: Float>(shape: &[usize], p: f64) -> ndarray::Array<T, ndarray::IxDyn> {
        ArrRng::default().bernoulli(shape, p)
    }

    /// Creates an ndarray sampled from a exponential distribution with given params.
    #[inline]
    pub fn exponential<T: Float>(
        shape: &[usize],
        lambda: f64,
    ) -> ndarray::Array<T, ndarray::IxDyn> {
        ArrRng::default().exponential(shape, lambda)
    }

    /// Creates an ndarray sampled from a log normal distribution with given params.
    #[inline]
    pub fn log_normal<T: Float>(
        shape: &[usize],
        mean: f64,
        stddev: f64,
    ) -> ndarray::Array<T, ndarray::IxDyn> {
        ArrRng::default().log_normal(shape, mean, stddev)
    }

    /// Creates an ndarray sampled from a gamma distribution with given params.
    #[inline]
    pub fn gamma<T: Float>(
        shape: &[usize],
        shape_param: f64,
        scale: f64,
    ) -> ndarray::Array<T, ndarray::IxDyn> {
        ArrRng::default().gamma(shape, shape_param, scale)
    }
}
