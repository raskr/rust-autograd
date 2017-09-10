extern crate rand;
extern crate ndarray;

use ndarray_ext::NdArray;
use self::rand::Rng;
use self::rand::distributions::IndependentSample;


#[inline]
fn generate_random<T>(shape: &[usize], dist: T) -> NdArray
    where T: IndependentSample<f64>
{
    let len = shape.iter().product();
    let samples: Vec<f32> = (0..len)
        .map(move |_| dist.ind_sample(&mut rand::thread_rng()) as f32)
        .collect();
    // unwrap is safe
    NdArray::from_shape_vec(shape, samples).unwrap()
}

#[inline]
fn generate_random_f<T, F>(shape: &[usize], dist: T, f: F) -> NdArray
where T: IndependentSample<f64>,
      F: Fn(f64) -> f64
{
    let len = shape.iter().sum();
    let samples: Vec<f32> = (0..len)
        .map(move |_| f(dist.ind_sample(&mut rand::thread_rng())) as f32)
        .collect();
    // unwrap is safe
    NdArray::from_shape_vec(shape, samples).unwrap()
}

#[inline]
/// Zeros.
pub fn zeros(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn> {
    NdArray::from_elem(shape, 0.)
}

#[inline]
/// Ones.
pub fn ones(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn> {
    NdArray::from_elem(shape, 1.)
}

#[inline]
/// Create ndarray object from a slice.
pub fn from_scalar(val: f32) -> ndarray::Array<f32, ndarray::IxDyn> {
    NdArray::from_elem(ndarray::IxDyn(&[1]), val)
}

#[inline]
/// Permutation.
pub fn permutation(size: usize) -> ndarray::Array1<usize> {
    let mut data: Vec<usize> = (0..size).collect();
    let slice = data.as_mut_slice();
    rand::thread_rng().shuffle(slice);
    ndarray::Array1::<usize>::from_vec(slice.to_vec())
}

#[inline]
/// Samples from normal distribution
pub fn random_normal(shape: &[usize], mean: f64, stddev: f64)
                     -> ndarray::Array<f32, ndarray::IxDyn> {
    let normal = rand::distributions::Normal::new(mean, stddev);
    generate_random(shape, normal)
}

#[inline]
/// Samples from uniform distribution.
pub fn random_uniform(shape: &[usize], min: f64, max: f64) -> ndarray::Array<f32, ndarray::IxDyn> {
    let range = rand::distributions::Range::new(min, max);
    generate_random(shape, range)
}

#[inline]
/// Samples from standard normal distribution
pub fn standard_normal(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn> {
    let normal = rand::distributions::Normal::new(0., 1.);
    generate_random(shape, normal)
}

#[inline]
/// Samples from standard normal distribution
pub fn standard_uniform(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn> {
    let dist = rand::distributions::Range::new(0., 1.);
    generate_random(shape, dist)
}

#[inline]
/// Glorot normal initialization. (a.k.a. Xavier normal initialization)
///
/// Normal distribution, but its scaling is determined by its input size.
pub fn glorot_normal(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn> {
    let s = 1. / (shape[0] as f64).sqrt();
    let normal = rand::distributions::Normal::new(0., s);
    generate_random(shape, normal)
}

#[inline]
/// Glorot uniform initialization. (a.k.a. Xavier uniform initialization)
///
/// Uniform distribution, but its range is determined by its input size.
pub fn glorot_uniform(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn> {
    let s = (6. / shape[0] as f64).sqrt();
    let uniform = rand::distributions::Range::new(-s, s);
    generate_random(shape, uniform)
}

/// Bernoulli distribution.
#[inline]
pub fn bernoulli(shape: &[usize], p: f64) -> ndarray::Array<f32, ndarray::IxDyn> {
    let dist = rand::distributions::Range::new(0., 1.);
    generate_random_f(shape, dist, |a| (a < p) as i64 as f64)
}

/// Exponential distribution.
#[inline]
pub fn exponential(shape: &[usize], lambda: f64) -> ndarray::Array<f32, ndarray::IxDyn> {
    let dist = rand::distributions::Exp::new(lambda);
    generate_random(shape, dist)
}

/// Log normal distribution.
#[inline]
pub fn log_normal(shape: &[usize], mean: f64, stddev: f64) -> ndarray::Array<f32, ndarray::IxDyn> {
    let dist = rand::distributions::LogNormal::new(mean, stddev);
    generate_random(shape, dist)
}

/// Gamma distribution.
#[inline]
pub fn gamma(shape: &[usize], shape_param: f64, scale: f64) -> ndarray::Array<f32, ndarray::IxDyn> {
    let dist = rand::distributions::Gamma::new(shape_param, scale);
    generate_random(shape, dist)
}
