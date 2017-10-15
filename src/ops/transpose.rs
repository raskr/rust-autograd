extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct Transpose {
    pub zip: bool,
}


impl ops::Op for Transpose {
    fn name(&self) -> &str
    {
        "Transpose"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0].view();
        let perm: &NdArray = &xs[1];
        assert!(perm.len() >= 2);


        if is_reversed(perm) {
            xs[0].clone().reversed_axes()
        } else {
            let src_dst = if self.zip {
                perm.iter()
                    .map(|&a| a as usize)
                    .zip(0..perm.len())
                    .collect::<Vec<_>>()
            } else {
                let mut a = perm.iter()
                    .map(|&a| a as usize)
                    .enumerate()
                    .collect::<Vec<_>>();
                a.sort_by_key(|sd| sd.1);
                a
            };

            do_transpose(x, src_dst)
        }
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let op = Transpose { zip: !self.zip };
        vec![Some(ops::apply_op(op, &[gy, inputs[1]])), None]
    }
}

fn do_transpose(mut x: ::ndarray_ext::NdArrayView, mut src_dst: Vec<(usize, usize)>) -> NdArray
{
    for i in 0..src_dst.len() {
        let (src, dst) = {
            let sd = src_dst[i];
            (sd.0, sd.1)
        };

        if src <= dst {
            continue;
        }

        for j in (dst..src).rev() {
            // "bigger to smaller" iteration is important
            x.swap_axes(j, j + 1); // Swaps two axes
            // Increments "src"es I passed by.
            for sd in src_dst.iter_mut() {
                if sd.0 == j {
                    sd.0 += 1;
                    break;
                }
            }
        }

        src_dst[i].0 = dst;
    }
    if x.is_standard_layout() {
        x.to_owned()
    } else {
        NdArray::from_shape_fn(x.shape(), |i| x[i])
    }

}

fn is_reversed(perm: &NdArray) -> bool
{
    use std::f32;
    let mut last = f32::MAX;
    for a in perm.iter() {
        if *a > last {
            return false;
        }
        last = *a
    }
    true
}
