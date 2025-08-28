use luminal::{op::InputTensor, prelude::*};
use ocl::{Buffer, ProQue};
pub mod prim;

/// Compile graphs to run on OpenCL-supported devices in supported data formats
pub type OpenCLCompiler<T> = OpenCLCompilerPreBuffer<T>;

/// All OpenCL compilers coming before buffer compilers
pub type OpenCLCompilerPreBuffer<T> = prim::PrimitiveCompiler<T>;

// TODO: impl data trait for buffer
#[derive(Debug, Clone)]
pub struct OpenCLBuffer(pub Buffer<f32>);

pub trait OpenCLKernel {
    fn opencl_forward();
}

#[macro_export]
macro_rules! debug_type {
    ($t: ident) => {
        impl std::fmt::Debug for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, stringify!($t))
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use luminal::prelude::*;

    use crate::OpenCLCompiler;

    #[test]
    fn cl_prim_test() {
        let mut cx = Graph::new();
        let a = cx.tensor(4).set([1.0, 2.0, 3.0, 4.0]);
        let b = cx.tensor(4).set([1.0, 2.0, 3.0, 4.0]);
        let mut c = (a + b).retrieve();

        cx.compile(OpenCLCompiler::<f16>::default(), &mut c);
        cx.execute();
        println!("{:?}", c);
    }
}
