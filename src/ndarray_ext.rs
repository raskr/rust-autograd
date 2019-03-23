use ndarray;
use Float;

pub type NdArray<T> = ndarray::Array<T, ndarray::IxDyn>;

pub type NdArrayView<'a, T> = ndarray::ArrayView<'a, T, ndarray::IxDyn>;

/// exposes array_gen
pub use array_gen::*;

#[inline]
pub fn arr_to_shape<T: Float>(arr: &NdArray<T>) -> Vec<usize> {
    arr.iter()
        .map(|&a| a.to_usize().unwrap())
        .collect::<Vec<_>>()
}

#[doc(hidden)]
#[inline]
pub fn expand_dims_view<T: Float>(x: NdArrayView<T>, axis: usize) -> NdArrayView<T> {
    let mut shape = x.shape().to_vec();
    shape.insert(axis, 1);
    x.into_shape(shape).unwrap()
}

#[doc(hidden)]
#[inline]
pub fn expand_dims<T: Float>(x: NdArray<T>, axis: usize) -> NdArray<T> {
    let mut shape = x.shape().to_vec();
    shape.insert(axis, 1);
    x.into_shape(shape).unwrap()
}

#[doc(hidden)]
#[inline]
pub fn roll_axis<T: Float>(arg: &mut NdArray<T>, to: ndarray::Axis, from: ndarray::Axis) {
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
pub fn normalize_negative_axis(axis: isize, ndim: usize) -> usize {
    if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    }
}

#[inline]
pub fn normalize_negative_axes<T: Float>(axes: &NdArray<T>, ndim: usize) -> Vec<usize> {
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
pub fn sparse_to_dense<T: Float>(arr: &NdArray<T>) -> Vec<usize> {
    let mut axes: Vec<usize> = vec![];
    for (i, &a) in arr.iter().enumerate() {
        if a == T::one() {
            axes.push(i);
        }
    }
    axes
}

#[doc(hidden)]
#[inline]
/// This works well only for small arrays
pub fn vec_as_shape<T: Float>(x: &NdArray<T>) -> Vec<usize> {
    let mut target = Vec::with_capacity(x.len());
    for &a in x.iter() {
        target.push(a.to_usize().unwrap());
    }
    target
}

#[doc(hidden)]
#[inline]
pub fn scalar_shape<T: Float>() -> NdArray<T> {
    // safe unwrap
    NdArray::from_shape_vec(ndarray::IxDyn(&[0]), vec![]).unwrap()
}

#[doc(hidden)]
#[inline]
pub fn is_scalar_shape(shape: &[usize]) -> bool {
    shape == &[] || shape == &[0]
}

#[doc(hidden)]
#[inline]
pub fn shape_of<T: Float>(x: &NdArray<T>) -> NdArray<T> {
    let shape = x
        .shape()
        .iter()
        .map(|&a| T::from(a).unwrap())
        .collect::<Vec<T>>();
    let rank = shape.len();
    // safe unwrap
    NdArray::from_shape_vec(ndarray::IxDyn(&[rank]), shape).unwrap()
}

#[doc(hidden)]
#[inline]
pub fn into_mat<T: Float>(x: NdArray<T>) -> ndarray::Array<T, ndarray::Ix2> {
    let (a, b) = {
        let shape = x.shape();
        (shape[0], shape[1])
    };
    x.into_shape(ndarray::Ix2(a, b)).unwrap()
}

/// Generates ndarrays which can be fed to `autograd::variable()` etc.
pub mod array_gen {
    use super::*;

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

use rand::distributions::IndependentSample;
use rand::{self, Rng, XorShiftRng};
use std::cell::RefCell;
use std::marker::PhantomData;

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
    pub fn new(rng: R) -> Self {
        ArrRng {
            phantom: PhantomData,
            rng: RefCell::new(rng),
        }
    }

    #[inline]
    fn alloc(size: usize) -> Vec<T> {
        let mut buf: Vec<T> = Vec::with_capacity(size);
        unsafe { buf.set_len(size) }
        buf
    }

    fn gen_random_array<I>(&self, shape: &[usize], dist: I) -> NdArray<T>
    where
        I: IndependentSample<f64>,
    {
        let size: usize = shape.into_iter().cloned().product();
        let mut buf = Self::alloc(size);
        let p = buf.as_mut_ptr();
        let mut rng = self.rng.borrow_mut();
        unsafe {
            for i in 0..size as isize {
                *p.offset(i) = T::from(dist.ind_sample(&mut *rng)).unwrap();
            }
        }
        NdArray::from_shape_vec(shape, buf).unwrap()
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
        let mut buf = Self::alloc(size);
        unsafe { buf.set_len(size) }
        let ptr = buf.as_mut_ptr();
        unsafe {
            for i in 0..size as isize {
                let val = dist.ind_sample(&mut *rng);
                *ptr.offset(i) = T::from(i32::from(val < p)).unwrap();
            }
        }
        i32::from(true);
        NdArray::from_shape_vec(shape, buf).unwrap()
    }

    pub fn exponential(&self, shape: &[usize], lambda: f64) -> ndarray::Array<T, ndarray::IxDyn> {
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
