extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct TransposeStatic {
    // This "must" be sorted by dst (second elem of the tuple)
    pub src_dst_sorted: Vec<(usize, usize)>,
}

pub struct TransposeDynamic {
    pub zip: bool,
}

pub struct ReverseAxes;

impl ops::Op for TransposeStatic {
    fn name(&self) -> &str
    {
        "TransposeStatic"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0].view();
        let src_dst = self.src_dst_sorted.clone();
        do_transpose(x, src_dst)
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let mut src_dst = self.src_dst_sorted
                              .iter()
            // swaps src and dst
                              .map(|sd| (sd.1, sd.0))
                              .collect::<Vec<_>>();

        // Sorts by dst. This forces all axes to move "right to left".
        src_dst.sort_by_key(|sd| sd.1);

        let op = TransposeStatic { src_dst_sorted: src_dst };

        vec![Some(ops::apply_op(op, &[gy]))]
    }
}

impl ops::Op for TransposeDynamic {
    fn name(&self) -> &str
    {
        "TransposeDynamic"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0].view();
        let perm: &NdArray = &xs[1];

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

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let op = TransposeDynamic { zip: !self.zip };
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
            // Swaps two axes
            x.swap_axes(j, j + 1);
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

impl ops::Op for ReverseAxes {
    fn name(&self) -> &str
    {
        "ReverseAxes"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        xs[0].clone().reversed_axes()
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(ops::transpose(gy, &[]))]
    }
}
