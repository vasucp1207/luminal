use ocl::{self, Buffer, OclPrm};
use std::{
    any::{Any, TypeId},
    fmt::{Debug, Write},
    ops::Deref,
    sync::Arc,
};

pub mod prim;
use prim::OpenCLConstant;

use itertools::Itertools;

use luminal::{op::InputTensor, prelude::*};

pub type OpenCLCompiler<T> = OpenCLCompilerPreBuffer<T>;

pub type OpenCLCompilerPreBuffer<T> = (prim::PrimitiveCompiler<T>,);

#[derive(Debug, Clone)]
pub struct OpenCLBuffer<T: OclPrm>(pub Buffer<T>);

impl<T: OclPrm> Deref for OpenCLBuffer<T> {
    type Target = Buffer<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: OclPrm> Data for OpenCLBuffer<T> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

pub trait OpenCLFloat: Copy + Debug + PartialEq + 'static + Default {
    fn to_f32(self) -> f32;
    fn from_f32(a: f32) -> Self;
    fn is_f32() -> bool;
    fn type_name() -> &'static str;
}

impl OpenCLFloat for f32 {
    fn from_f32(a: f32) -> Self {
        a
    }
    fn to_f32(self) -> f32 {
        self
    }
    fn is_f32() -> bool {
        true
    }
    fn type_name() -> &'static str {
        "float"
    }
}

impl OpenCLFloat for f16 {
    fn from_f32(a: f32) -> Self {
        f16::from_f32(a)
    }
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    fn is_f32() -> bool {
        false
    }
    fn type_name() -> &'static str {
        "half"
    }
}

pub trait OpenCLKernel<T: OclPrm>: Debug {
    /// Annotate the buffer sizes of the intermediate buffers
    fn intermediate_buffer_sizes(&self, _: &[ShapeTracker]) -> Vec<Expression> {
        vec![]
    }
    /// Annotate the buffer sizes of the output buffers
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<Expression>;
    /// Set up the kernel on the buffer
    fn opencl_forward(
        &mut self,
        inputs: &[(&Buffer<T>, ShapeTracker)],
        intermediate_buffers: &[&Buffer<T>],
        output_buffers: &[&Buffer<T>],
    );
}

#[derive(Clone)]
pub struct OpenCLKernelWrapper<T: OclPrm>(pub Arc<Box<dyn OpenCLKernel<T>>>);
impl<T: OclPrm> Debug for OpenCLKernelWrapper<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OpenCLKernelWrapper")
    }
}

impl<T: OclPrm> Default for OpenCLKernelWrapper<T> {
    fn default() -> Self {
        Self(Arc::new(Box::new(())))
    }
}

impl<T: OclPrm> Deref for OpenCLKernelWrapper<T> {
    type Target = Box<dyn OpenCLKernel<T>>;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl<T: OclPrm> OpenCLKernel<T> for () {
    fn output_buffer_sizes(&self, _: &[ShapeTracker]) -> Vec<Expression> {
        vec![]
    }
    fn opencl_forward(
        &mut self,
        _: &[(&Buffer<T>, ShapeTracker)],
        _: &[&Buffer<T>],
        _: &[&Buffer<T>],
    ) {
    }
}

fn is<T: Any>(type_id: TypeId) -> bool {
    type_id == TypeId::of::<T>()
}

fn expr_to_opencl_string(expr: &Expression) -> String {
    let mut symbols = vec![];
    for term in expr.terms.read().clone() {
        let new_symbol = match term {
            Term::Num(n) => n.to_string(),
            Term::Var(c) => {
                if c == 'z' {
                    "(int)idx".to_string()
                } else {
                    c.to_string()
                }
            }
            Term::Max => format!(
                "max((int){}, (int){})",
                symbols.pop().unwrap(),
                symbols.pop().unwrap()
            ),
            Term::Min => format!(
                "min((int){}, (int){})",
                symbols.pop().unwrap(),
                symbols.pop().unwrap()
            ),
            Term::Lt => format!(
                "(int)({} < {})",
                symbols.pop().unwrap(),
                symbols.pop().unwrap()
            ),
            Term::Gte => format!(
                "(int)({} >= {})",
                symbols.pop().unwrap(),
                symbols.pop().unwrap()
            ),
            _ => format!(
                "({}{term:?}{})",
                symbols.pop().unwrap(),
                symbols.pop().unwrap()
            ),
        };
        symbols.push(new_symbol);
    }
    symbols.pop().unwrap()
}

fn get_idx_valid_exps(shape: ShapeTracker) -> (String, String) {
    (
        expr_to_opencl_string(&shape.index_expression()),
        expr_to_opencl_string(&shape.valid_expression()),
    )
}

fn get_buffer_from_tensor<'a, T: OclPrm>(tensor: &'a InputTensor) -> &'a OpenCLBuffer<T> {
    tensor
        .borrowed()
        .downcast_ref::<OpenCLBuffer<T>>()
        .expect("Tensor does not contain a opencl buffer")
}

pub fn constant<T: OpenCLFloat + OclPrm>(num: f32) -> SelectGraph {
    let mut n = op::<OpenCLConstant<T>>();
    n.check(move |o, _| {
        if let Some(c) = o.as_any().downcast_ref::<OpenCLConstant<T>>() {
            if let luminal::op::ConstantValue::Float(f) = c.0 {
                (f - num).abs() < 1e-3
            } else {
                false
            }
        } else {
            false
        }
    });
    n
}

fn render_dyn_dim_inputs(shapes: &[ShapeTracker], offset: usize) -> (Vec<char>, String) {
    let symbols: Vec<char> = shapes
        .iter()
        .flat_map(|st| {
            st.dims()
                .into_iter()
                .chain(st.padding.into_iter().flat_map(|i| [i.0, i.1]))
                .chain(st.mask.into_iter().flat_map(|i| [i.0, i.1]))
        })
        .flat_map(|d| d.to_symbols())
        .unique()
        .collect();
    (
        symbols.clone(),
        symbols
            .into_iter()
            .enumerate()
            .fold(String::default(), |mut acc, (i, c)| {
                write!(&mut acc, ", device int& {c} [[buffer({})]]", i + offset).unwrap();
                acc
            }),
    )
}

#[macro_export]
macro_rules! debug_type {
    ($t: ident) => {
        impl<T> std::fmt::Debug for $t<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, stringify!($t))
            }
        }
    };
}
