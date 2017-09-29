extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct Transpose {
    // This "must" be sorted by dst (second elem of the tuple)
    pub src_dst_sorted: Vec<(usize, usize)>,
}

impl ops::Op for Transpose {
    fn name(&self) -> &str
    {
        "Transpose"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let mut x = xs[0].view();

        let mut src_dst = self.src_dst_sorted.clone();

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

    fn grad(&self, gy: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let mut src_dst = self.src_dst_sorted
            .iter()
            // swaps src and dst
            .map(|sd| (sd.1, sd.0))
            .collect::<Vec<_>>();

        // Sorts by dst. This forces all axes to move "right to left".
        src_dst.sort_by_key(|sd| sd.1);

        let op = Transpose { src_dst_sorted: src_dst };

        vec![Some(ops::apply_op(op, &[gy]))]
    }
}
