use crate::OpenCLKernel;
use luminal::prelude::Function as LFunction;
use luminal::prelude::*;
use ocl::{ProQue, builders::ProQueBuilder};
use std::marker::PhantomData;

/// Copy a tensor to the GPU
#[derive(Clone)]
pub struct OpenCLCopyToDevice(ProQue);
crate::debug_type!(OpenCLCopyToDevice);

impl OpenCLCopyToDevice {
    fn new() {
        todo!()
    }
}

impl Operator for OpenCLCopyToDevice {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        todo!()
    }
}

/// Copy a tensor from the GPU
#[derive(Clone)]
pub struct OpenCLFromDevice(ProQue);
crate::debug_type!(OpenCLFromDevice);

impl OpenCLFromDevice {
    fn new() {
        todo!()
    }
}

impl Operator for OpenCLFromDevice {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        todo!()
    }
}

#[macro_export]
macro_rules! opencl_binary_op {
    ($fn:expr, $name:ident) => {
        #[derive(Clone)]
        pub struct $name {
            pro_que: ProQue,
        }
        $crate::debug_type!($name);

        impl $name {
            pub fn new() {
                let code = format!(
                    "
kernel void clkernel() {{

}}
                    "
                );
            }
        }

        impl OpenCLKernel for $name {
            fn opencl_forward() {
                // TODO: pipeline setting
                todo!()
            }
        }

        impl Operator for $name {
            fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
                // TODO: this will call forward
                todo!()
            }
        }
    };
}

opencl_binary_op!(|a, b| format!("{a} + {b}"), OpenCLAdd);

#[derive(Default, Debug)]
pub struct PrimitiveCompiler<T>(PhantomData<T>);

impl<T> Compiler for PrimitiveCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let pro_que = ProQue::builder();
        // Iterate the graph
        for function_node in graph
            .node_indices()
            .filter(|n| graph.node_weight(*n).unwrap().as_any().is::<LFunction>())
            .collect::<Vec<_>>()
        {
            // Copy outputs to device

            // Insert copy from device for function inputs

            // Copy to_retrieve from device

            // Swap primitive ops
            for id in graph.node_indices().collect::<Vec<_>>() {
                let op: NodeIndex;
                let op_ref: NodeIndex;
            }
        }
    }
}
