use crate::op;
use crate::Float;
use std::marker::PhantomData;

pub(crate) struct HookOp<T: Float, H: crate::hook::Hook<T>> {
    phantom: PhantomData<T>,
    pub hook: H,
}

impl<T: Float, H: crate::hook::Hook<T>> HookOp<T, H> {
    #[inline]
    pub fn new(hook: H) -> Self {
        HookOp {
            phantom: PhantomData,
            hook,
        }
    }
}

impl<T: Float, H: crate::hook::Hook<T>> op::Op<T> for HookOp<T, H> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0);
        self.hook.call(&ret);
        ctx.append_output_view(ret);
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(Some(ctx.output_grad()));
    }
}
