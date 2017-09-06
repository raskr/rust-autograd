extern crate rand;
extern crate ndarray;

use ndarray_ext::NdArray;
use self::rand::Rng;
use self::rand::distributions::IndependentSample;


/// Glorot normal initialization. (a.k.a. Xavier normal initialization)
///
/// Normal distribution, but its scaling is determined by its input size.
pub fn glorot_normal(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn> {
    let mut default = NdArray::default(shape);
    let s = 1. / (default.shape()[0] as f64).sqrt();
    let normal = rand::distributions::Normal::new(0., s);

    for elem in default.iter_mut() {
        *elem = normal.ind_sample(&mut rand::thread_rng()) as f32;
    }
    default
}

/// Glorot uniform initialization. (a.k.a. Xavier uniform initialization)
///
/// Uniform distribution, but its range is determined by its input size.
pub fn glorot_uniform(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn> {
    let mut default = NdArray::default(shape);
    let s = (6. / default.shape()[0] as f64).sqrt();
    let uniform = rand::distributions::Range::new(-s, s);

    for elem in default.iter_mut() {
        *elem = uniform.ind_sample(&mut rand::thread_rng()) as f32;
    }
    default
}

/// Normal distribution.
pub fn randn(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn> {
    let mut default = NdArray::default(shape);
    let normal = rand::distributions::Normal::new(0., 1.);

    for elem in default.iter_mut() {
        *elem = normal.ind_sample(&mut rand::thread_rng()) as f32;
    }
    default
}

/// Uniform distribution.
pub fn randu(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn> {
    let mut default = NdArray::default(shape);
    let uniform = rand::distributions::Range::new(0., 1.);

    for elem in default.iter_mut() {
        *elem = uniform.ind_sample(&mut rand::thread_rng()) as f32;
    }
    default
}

/// Zeros.
pub fn zeros(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn> {
    NdArray::from_elem(shape, 0.)
}

/// Ones.
pub fn ones(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn> {
    NdArray::from_elem(shape, 1.)
}

/// Create ndarray object from a slice.
pub fn from_slice(slice: &[f32], shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn> {
    if let Ok(a) = NdArray::from_shape_vec(ndarray::IxDyn(shape), slice.to_vec()) {
        a
    } else {
        panic!("can't reshape input slice to shape you provided ");
    }
}

/// Create ndarray object from a slice.
pub fn from_scalar(val: f32) -> ndarray::Array<f32, ndarray::IxDyn> {
    NdArray::from_elem(ndarray::IxDyn(&[1]), val)
}

/// Create ndarray object from a Vec.
pub fn from_vec(vec: Vec<f32>, shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn> {
    if let Ok(a) = NdArray::from_shape_vec(ndarray::IxDyn(shape), vec) {
        a
    } else {
        panic!("can't reshape input slice to shape you provided ");
    }
}

/// Permutation.
#[inline]
pub fn permutation(size: usize) -> ndarray::Array1<usize> {
    let mut data: Vec<usize> = (0..size).collect();
    let slice = data.as_mut_slice();
    rand::thread_rng().shuffle(slice);
    ndarray::Array1::<usize>::from_vec(slice.to_vec())
}
