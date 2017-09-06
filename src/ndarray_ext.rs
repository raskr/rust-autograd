/// small extension of rust-ndarray for convenience
extern crate ndarray;

use std::f32;


/// type alias for convenience
pub type NdArray = ndarray::Array<f32, ndarray::IxDyn>;
pub type NdArrayView<'a> = ndarray::ArrayView<'a, f32, ndarray::IxDyn>;


#[inline]
// TODO: remove unwrap
pub fn expand_dims_view<'a>(x: NdArrayView<'a>, axis: usize) -> NdArrayView<'a> {
    let mut shape = x.shape().to_vec();
    shape.insert(axis, 1);
    x.into_shape(shape).unwrap()
}

#[inline]
// TODO: remove unwrap
pub fn expand_dims(x: NdArray, axis: usize) -> NdArray {
    let mut shape = x.shape().to_vec();
    shape.insert(axis, 1);
    x.into_shape(shape).unwrap()
}

#[inline]
pub fn roll_axis(arg: &mut NdArray, to: ndarray::Axis, from: ndarray::Axis) {
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
pub fn into_mat(x: NdArray) -> ndarray::Array<f32, ndarray::Ix2> {
    let a = x.shape()[0];
    let b = x.shape()[1];
    x.into_shape(ndarray::Ix2(a, b)).unwrap()
}


#[test]
pub fn bbb() {
    let arr = ::init::zeros(&[1, 2, 3]);
    let batch_size = 100;
    let num_samples = 60000;
    let num_batches = num_samples % batch_size;
    let perm = ::init::permutation(num_samples).to_vec();

    for i in 0..num_batches {
        let x_indices = perm[i..i+num_batches].to_vec();
        let x_batch = arr.select(ndarray::Axis(0), x_indices.as_slice());
    }
}
