use crate::ndarray_ext::NdArrayView;
use crate::op;
use crate::tensor::Tensor;
use crate::Float;

pub struct Hook<T: Float> {
    pub name: Option<String>,
    pub func: Box<Fn(&NdArrayView<T>) -> ()>,
}

impl<T: Float> op::Op<T> for Hook<T> {
    fn name(&self) -> &str {
        "Hook"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let ret = ctx.grab_inputs()[0].clone();
        if let Some(ref a) = self.name {
            println!("{}:", a);
        }
        (self.func)(&ret);
        vec![Ok(crate::ArrRepr::View(ret))]
    }

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(gy.clone())]
    }
}
