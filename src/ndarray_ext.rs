/// small extension of rust-ndarray for convenience
extern crate ndarray;

/// type alias for convenience
pub type NdArray = ndarray::Array<f32, ndarray::IxDyn>;

/// type alias for convenience
pub type NdArrayView<'a> = ndarray::ArrayView<'a, f32, ndarray::IxDyn>;

/// exposes array_gen
pub use array_gen::*;



#[inline]
pub fn arr_to_shape(arr: &NdArray) -> Vec<usize>
{
    arr.iter().map(|&a| a as usize).collect::<Vec<_>>()
}


#[doc(hidden)]
#[inline]
// TODO: remove unwrap
pub fn expand_dims_view<'a>(x: NdArrayView<'a>, axis: usize) -> NdArrayView<'a>
{
    let mut shape = x.shape().to_vec();
    shape.insert(axis, 1);
    x.into_shape(shape).unwrap()
}

#[doc(hidden)]
#[inline]
// TODO: remove unwrap
pub fn expand_dims(x: NdArray, axis: usize) -> NdArray
{
    let mut shape = x.shape().to_vec();
    shape.insert(axis, 1);
    x.into_shape(shape).unwrap()
}

#[doc(hidden)]
#[inline]
pub fn roll_axis(arg: &mut NdArray, to: ndarray::Axis, from: ndarray::Axis)
{
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
#[doc(hidden)]
pub fn axes_as_vec(axes_: &NdArray, ndim: usize, sparse_axes: bool) -> Vec<usize>
{
    if sparse_axes {
        let mut axes: Vec<usize> = vec![];
        for (i, &a) in axes_.iter().enumerate() {
            if a == 1. {
                axes.push(i as usize);
            }
        }
        axes
    } else {
        let mut axes: Vec<usize> = Vec::with_capacity(axes_.len());
        for &axis in axes_.iter() {
            let axis = if axis < 0. {
                (ndim as f32 + axis) as usize
            } else {
                axis as usize
            };
            axes.push(axis);
        }
        axes
    }
}


#[doc(hidden)]
pub fn broadcast_to(
    broadcast_objective: &NdArray,
    target: &[usize],
    reduction_axes: &NdArray,
    keep_dims: bool,
    sparse_axes: bool,
) -> NdArray
{
    let mut gy = broadcast_objective.view();

    // convert axes_ to usize vec
    let mut axes = axes_as_vec(reduction_axes, target.len(), sparse_axes);

    // make broadcast dims
    if !keep_dims {
        axes.sort();
        for &axis in axes.iter() {
            gy = expand_dims_view(gy, axis);
        }
    }

    // do broadcast
    if let Some(gx) = gy.broadcast(target) {
        gx.to_owned()
    } else {
        panic!("Broadcast failed. This is probably a bug.")
    }
}

#[doc(hidden)]
#[inline]
/// This works well only for small array
pub fn vec_as_shape(x: &NdArray) -> Vec<usize>
{
    let mut target = Vec::with_capacity(x.len());
    for &a in x.iter() {
        target.push(a as usize);
    }
    target
}

#[doc(hidden)]
#[inline]
pub fn scalar_shape() -> NdArray
{
    // safe unwrap
    NdArray::from_shape_vec(ndarray::IxDyn(&[0]), vec![]).unwrap()
}

#[doc(hidden)]
#[inline]
pub fn shape_of(x: &NdArray) -> NdArray
{
    let shape = x.shape().iter().map(|&a| a as f32).collect::<Vec<f32>>();
    let rank = shape.len();
    // safe unwrap
    NdArray::from_shape_vec(ndarray::IxDyn(&[rank]), shape).unwrap()
}


#[doc(hidden)]
#[inline]
pub fn into_mat(x: NdArray) -> ndarray::Array<f32, ndarray::Ix2>
{
    let (a, b) = {
        let shape = x.shape();
        (shape[0], shape[1])
    };
    x.into_shape(ndarray::Ix2(a, b)).unwrap()
}


/// Generates ndarray which can be fed to `autograd::variable()` etc.
pub mod array_gen {
    extern crate rand;
    extern crate ndarray;

    // `Rng` trait must be included

    use self::rand::Rng;
    use self::rand::distributions::IndependentSample;
    use ndarray_ext::NdArray;

    #[inline]
    fn gen_rnd_array<T>(shape: &[usize], dist: T) -> NdArray
    where
        T: IndependentSample<f64>,
    {
        //        let mut rng = XorShiftRng::new_unseeded();
        let mut rng = rand::weak_rng();
        NdArray::from_shape_fn(shape, |_| dist.ind_sample(&mut rng) as f32)
    }

    #[inline]
    fn gen_rand_array_f<T, F>(shape: &[usize], dist: T, f: F) -> NdArray
    where
        T: IndependentSample<f64>,
        F: Fn(f64) -> f64,
    {
        //        let mut rng = XorShiftRng::new_unseeded();
        let mut rng = rand::weak_rng();
        NdArray::from_shape_fn(shape, |_| f(dist.ind_sample(&mut rng)) as f32)
    }

    #[inline]
    /// Zeros.
    pub fn zeros(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        NdArray::from_elem(shape, 0.)
    }

    #[inline]
    /// Ones.
    pub fn ones(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        NdArray::from_elem(shape, 1.)
    }

    #[inline]
    /// Create ndarray object from a scalar.
    pub fn from_scalar(val: f32) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        NdArray::from_elem(ndarray::IxDyn(&[]), val)
    }

    #[inline]
    /// Permutation.
    pub fn permutation(size: usize) -> ndarray::Array1<usize>
    {
        let mut data: Vec<usize> = (0..size).collect();
        let slice = data.as_mut_slice();

        rand::weak_rng().shuffle(slice);
        ndarray::Array1::<usize>::from_vec(slice.to_vec())
    }

    #[inline]
    /// Samples from normal distribution
    pub fn random_normal(
        shape: &[usize],
        mean: f64,
        stddev: f64,
    ) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        let normal = rand::distributions::Normal::new(mean, stddev);
        gen_rnd_array(shape, normal)
    }

    #[inline]
    /// Samples from uniform distribution.
    pub fn random_uniform(
        shape: &[usize],
        min: f64,
        max: f64,
    ) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        let range = rand::distributions::Range::new(min, max);
        gen_rnd_array(shape, range)
    }

    #[inline]
    /// Samples from standard normal distribution
    pub fn standard_normal(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        let normal = rand::distributions::Normal::new(0., 1.);
        gen_rnd_array(shape, normal)
    }

    #[inline]
    /// Samples from standard uniform distribution
    pub fn standard_uniform(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        let dist = rand::distributions::Range::new(0., 1.);
        gen_rnd_array(shape, dist)
    }

    #[inline]
    /// Glorot normal initialization. (a.k.a. Xavier normal initialization)
    pub fn glorot_normal(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        assert_eq!(shape.len(), 2);
        let s = 1. / (shape[0] as f64).sqrt();
        let normal = rand::distributions::Normal::new(0., s);
        gen_rnd_array(shape, normal)
    }

    #[inline]
    /// Glorot uniform initialization. (a.k.a. Xavier uniform initialization)
    pub fn glorot_uniform(shape: &[usize]) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        assert_eq!(shape.len(), 2);
        let s = (6. / shape[0] as f64).sqrt();
        let uniform = rand::distributions::Range::new(-s, s);
        gen_rnd_array(shape, uniform)
    }

    /// Bernoulli distribution.
    #[inline]
    pub fn bernoulli(shape: &[usize], p: f64) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        let dist = rand::distributions::Range::new(0., 1.);
        gen_rand_array_f(shape, dist, |a| (a < p) as i64 as f64)
    }

    /// Exponential distribution.
    #[inline]
    pub fn exponential(shape: &[usize], lambda: f64) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        let dist = rand::distributions::Exp::new(lambda);
        gen_rnd_array(shape, dist)
    }

    /// Log normal distribution.
    #[inline]
    pub fn log_normal(
        shape: &[usize],
        mean: f64,
        stddev: f64,
    ) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        let dist = rand::distributions::LogNormal::new(mean, stddev);
        gen_rnd_array(shape, dist)
    }

    /// Gamma distribution.
    #[inline]
    pub fn gamma(
        shape: &[usize],
        shape_param: f64,
        scale: f64,
    ) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        let dist = rand::distributions::Gamma::new(shape_param, scale);
        gen_rnd_array(shape, dist)
    }
}
