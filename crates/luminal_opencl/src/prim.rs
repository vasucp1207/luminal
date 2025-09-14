use std::{any::Any, fmt::Debug, marker::PhantomData, mem::size_of};

use super::*;
use ocl::{
    Buffer, Context, Device, Kernel, OclPrm, Platform, ProQue, Program, Queue, SpatialDims,
    builders::ProgramBuilder,
};
use petgraph::visit::EdgeRef;
use rustc_hash::FxHashMap;

use luminal::op::Function as LFunction;

/// Copy a tensor to the GPU
#[derive(Clone)]
pub struct OpenCLCopyToDevice<T>(Queue, PhantomData<T>);
crate::debug_type!(OpenCLCopyToDevice);

impl<T> OpenCLCopyToDevice<T> {
    pub fn new(queue: Queue) -> Self {
        Self(queue, PhantomData)
    }
}

impl<T: OpenCLFloat + OclPrm> Operator for OpenCLCopyToDevice<T> {
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().is::<OpenCLBuffer<T>>() {
            // Already on device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let mut data = inp[0]
            .0
            .borrowed()
            .downcast_ref::<Vec<f32>>()
            .unwrap()
            .iter()
            .copied()
            .map(OpenCLFloat::from_f32)
            .collect::<Vec<T>>();
        if data.is_empty() {
            data.push(T::from_f32(0.0));
        }
        let buffer = Buffer::<T>::builder()
            .queue(self.0.clone())
            .len(data.len())
            .fill_val(Default::default())
            .build()
            .unwrap();
        buffer.write(&data).enq().unwrap();
        vec![Tensor::new(OpenCLBuffer(buffer))]
    }
}

/// Copy a tensor from the GPU
#[derive(Clone, Default)]
pub struct OpenCLCopyFromDevice<T>(PhantomData<T>);
crate::debug_type!(OpenCLCopyFromDevice);

impl<T: OpenCLFloat + OclPrm> Operator for OpenCLCopyFromDevice<T> {
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().is::<Vec<f32>>() {
            // Already off device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let buffer = get_buffer_from_tensor(&inp[0].0);
        let mut data = vec![T::default(); buffer.len() as usize / std::mem::size_of::<T>()];
        if data.is_empty() {
            data.push(T::from_f32(0.0));
        }
        buffer.0.read(&mut data).enq().unwrap();
        let out: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
        vec![Tensor::new(out)]
    }
}

#[derive(Clone)]
pub struct OpenCLConstant<T>(
    pub ConstantValue,
    pub Queue,
    pub *const FxHashMap<char, usize>,
    pub PhantomData<T>,
);

impl<T> PartialEq for OpenCLConstant<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T> Debug for OpenCLConstant<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OpenCLConstant({:?})", self.0)
    }
}

impl<T: OpenCLFloat + OclPrm> Operator for OpenCLConstant<T> {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let val = T::from_f32(match &self.0 {
            ConstantValue::Expression(e) => {
                e.exec(unsafe { self.2.as_ref().unwrap() }).unwrap() as f32
            }
            ConstantValue::Float(f) => *f,
        });
        let buffer = Buffer::<T>::builder()
            .queue(self.1.clone())
            .len(1) // usize or u64
            .fill_val(Default::default())
            .build()
            .unwrap();

        buffer.write(&[val][..]).enq().unwrap();
        vec![Tensor::new(OpenCLBuffer(buffer))]
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "elementwise" {
            if let ConstantValue::Float(f) = self.0 {
                return Some(Box::new(format!("{f:?}")));
            }
        }
        None
    }
}

#[macro_export]
macro_rules! opencl_unary_op {
    ($op: expr, $op_name: ident) => {
        #[derive(Clone)]
        pub struct $op_name<T> {
            proque: ProQue,
            queue: Queue,
            kernel: Option<Arc<Kernel>>,
            dyn_symbols: Vec<char>,
            dyn_map: *const FxHashMap<char, usize>,
            _phantom: PhantomData<T>,
        }
        $crate::debug_type!($op_name);

        impl<T: OpenCLFloat + OclPrm> $op_name<T> {
            pub fn new(
                shape: ShapeTracker,
                device: Device,
                context: Context,
                queue: Queue,
                dyn_map: *const FxHashMap<char, usize>,
            ) -> Self {
                let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
                let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape], 3);
                let type_name = T::type_name();
                let code = format!(
                    "
__kernel void clkernel(const __global {type_name}* a, __global {type_name}* out, const int n_elements{rendered}) {{
    int idx = get_global_id(0);
    if (idx < n_elements && {valid_exp} != 0) {{
        out[idx] = {}(a[{idx_exp}]);
    }}
}}
                ",
                    $op
                );

                let mut pb = ProgramBuilder::new();
                pb.src(code);
                let src_str = pb.get_src_strings().map_err(|e| e.to_string()).unwrap();
                let cmplr_opts = pb
                    .get_compiler_options()
                    .map_err(|e| e.to_string())
                    .unwrap();
                let program =
                    Program::with_source(&context, &src_str, Some(&[device]), &cmplr_opts).unwrap();
                let dims = SpatialDims::One(1);

                let proque = ProQue::new(context.clone(), queue.clone(), program, Some(dims));
                Self {
                    proque,
                    queue,
                    kernel: None,
                    dyn_symbols,
                    dyn_map,
                    _phantom: Default::default(),
                }
            }
        }

        impl<T: OclPrm> OpenCLKernel<T> for $op_name<T> {
            fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<Expression> {
                vec![input_shapes[0].contiguous().n_elements() * size_of::<T>()]
            }
            fn opencl_forward(
                &mut self,
                inputs: &[(&Buffer<T>, ShapeTracker)],
                _: &[&Buffer<T>],
                output_buffers: &[&Buffer<T>],
            ) {
                let inp_size = inputs[0].1.n_elements().to_usize().unwrap();
                let mut builder = self.proque.kernel_builder("clkernel");
                builder
                    .arg(inputs[0].0)
                    .arg(output_buffers[0])
                    .arg(inp_size as u32);

                unsafe {
                    for s in &self.dyn_symbols {
                        let val = self.dyn_map.as_ref().unwrap()[s] as u32;
                        builder.arg(val);
                    }
                }

                let kernel = builder.build().unwrap();

                unsafe {
                    kernel.cmd().global_work_size(inp_size).enq().unwrap();
                }

                self.kernel = Some(Arc::new(kernel));
            }
        }

        impl<T: OpenCLFloat + OclPrm> Operator for $op_name<T> {
            fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
                let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
                let out = Buffer::<T>::builder()
                    .queue(self.queue.clone())
                    .len((inp_size * size_of::<T>()) as usize)
                    .fill_val(Default::default())
                    .build()
                    .unwrap();

                self.opencl_forward(
                    &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
                    &[],
                    &[&out],
                );

                unsafe {
                    self.kernel.as_ref().unwrap().enq().unwrap();
                }
                vec![Tensor::new(OpenCLBuffer(out))]
            }

            fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
                if key == "opencl" {
                    return Some(Box::new(OpenCLKernelWrapper(Arc::new(Box::new(
                        self.clone(),
                    )))));
                }
                if key == "elementwise" {
                    return Some(Box::new(format!("{}(input0)", $op)));
                }
                None
            }
        }
    };
}

opencl_unary_op!("", OpenCLContiguous);
opencl_unary_op!("log2", OpenCLLog2);
opencl_unary_op!("exp2", OpenCLExp2);
opencl_unary_op!("sin", OpenCLSin);
opencl_unary_op!("sqrt", OpenCLSqrt);
opencl_unary_op!("1.0 / ", OpenCLRecip);

#[macro_export]
macro_rules! opencl_binary_op {
    ($fn:expr, $name:ident) => {
        #[derive(Clone)]
        pub struct $name<T> {
            proque: ProQue,
            queue: Queue,
            kernel: Option<Arc<Kernel>>,
            _phantom: PhantomData<T>,
            dyn_symbols: Vec<char>,
            dyn_map: *const FxHashMap<char, usize>,
        }
        $crate::debug_type!($name);

        impl<T: OpenCLFloat + OclPrm> $name<T> {
            pub fn new(
                a_shape: ShapeTracker,
                b_shape: ShapeTracker,
                device: Device,
                context: Context,
                queue: Queue,
                dyn_map: *const FxHashMap<char, usize>,
            ) -> Self {
                let (a_idx_exp, a_valid_exp) = get_idx_valid_exps(a_shape);
                let (b_idx_exp, b_valid_exp) = get_idx_valid_exps(b_shape);
                let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[a_shape, b_shape], 4);
                let type_name = T::type_name();
                let op = $fn(
                    format!("(({}) == 0 ? 0.0h : a[{}])", a_valid_exp, a_idx_exp),
                    format!("(({}) == 0 ? 0.0h : b[{}])", b_valid_exp, b_idx_exp)
                );

                let code = format!(
                    "
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void clkernel(const __global {type_name}* a, __global {type_name}* b, __global {type_name}* out, const int n_elements{rendered}) {{
    int idx = get_global_id(0);
    if (idx < n_elements) {{
        out[idx] = {op};
    }}
}}
                "
                );

                let mut pb = ProgramBuilder::new();
                pb.src(code);
                let src_str = pb.get_src_strings().map_err(|e| e.to_string()).unwrap();
                let cmplr_opts = pb
                    .get_compiler_options()
                    .map_err(|e| e.to_string())
                    .unwrap();
                let program =
                    Program::with_source(&context, &src_str, Some(&[device]), &cmplr_opts).unwrap();
                // fake dim
                let dims = SpatialDims::One(1);
                let proque = ProQue::new(context.clone(), queue.clone(), program, Some(dims));
                Self {
                    proque,
                    queue,
                    kernel: None,
                    _phantom: Default::default(),
                    dyn_symbols,
                    dyn_map,
                }
            }
        }

        impl<T: OclPrm> OpenCLKernel<T> for $name<T> {
            fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<Expression> {
                vec![input_shapes[0].n_elements() * size_of::<T>()]
            }
            fn opencl_forward(
                &mut self,
                inputs: &[(&Buffer<T>, ShapeTracker)],
                _: &[&Buffer<T>],
                output_buffers: &[&Buffer<T>],
            ) {
                let inp_size = inputs[0].1.n_elements().to_usize().unwrap();
                let mut builder = self.proque.kernel_builder("clkernel");
                builder
                    .arg(inputs[0].0)
                    .arg(inputs[1].0)
                    .arg(output_buffers[0])
                    .arg(inp_size as u32);

                unsafe {
                    for s in &self.dyn_symbols {
                        let val = self.dyn_map.as_ref().unwrap()[s] as u32;
                        builder.arg(val);
                    }
                }

                let kernel = builder.build().unwrap();

                unsafe {
                    kernel.cmd().global_work_size(inp_size).enq().unwrap();
                }

                self.kernel = Some(Arc::new(kernel));
            }
        }

        impl<T: OpenCLFloat + OclPrm> Operator for $name<T> {
            fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
                let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
                let out = Buffer::<T>::builder()
                    .queue(self.queue.clone())
                    .len((inp_size * size_of::<T>()) as usize) // usize or u64
                    .fill_val(Default::default())
                    .build()
                    .unwrap();

                self.opencl_forward(
                    &[
                        (get_buffer_from_tensor(&tensors[0].0), tensors[0].1),
                        (get_buffer_from_tensor(&tensors[1].0), tensors[1].1),
                    ],
                    &[],
                    &[&out],
                );

                unsafe {
                    self.kernel.as_ref().unwrap().enq().unwrap();
                }
                vec![Tensor::new(OpenCLBuffer(out))]
            }

            fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
                if key == "opencl" {
                    return Some(Box::new(OpenCLKernelWrapper(Arc::new(Box::new(
                        self.clone(),
                    )))));
                }
                if key == "elementwise" {
                    return Some(Box::new($fn("input0", "input1")));
                }
                None
            }
        }
    };
}

opencl_binary_op!(|a, b| format!("{a} + {b}"), OpenCLAdd);
opencl_binary_op!(|a, b| format!("{a} * {b}"), OpenCLMul);
opencl_binary_op!(|a, b| format!("(float)({a} < {b})"), OpenCLLessThan);
opencl_binary_op!(|a, b| format!("fmod({a}, {b})"), OpenCLMod);

#[derive(Clone)]
pub struct OpenCLSumReduce<T> {
    proque: ProQue,
    queue: Queue,
    kernel: Option<Arc<Kernel>>,
    pub dim: usize,
    pub shape: ShapeTracker,
    dyn_symbols: Vec<char>,
    _phantom: PhantomData<T>,
    dyn_map: *const FxHashMap<char, usize>,
}
crate::debug_type!(OpenCLSumReduce);

impl<T> PartialEq for OpenCLSumReduce<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
    }
}

impl<T: OpenCLFloat> OpenCLSumReduce<T> {
    pub fn new(
        shape: ShapeTracker,
        dim: usize,
        device: Device,
        context: Context,
        queue: Queue,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape], 6);
        let type_name = T::type_name();
        let code = format!(
            "
__kernel void clkernel(const __global float* a, __global float* out, const int n_elements, const int front_size, const int back_size, const int dim_size{rendered}) {{
    int i = get_global_id(0);
    if (i < n_elements) {{
        int a_ = i / back_size;
        int b_ = i % back_size;
        float reduce_value = 0.0;
        for (int c_ = 0; c_ < dim_size; c_++) {{
            uint idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid_exp}) != 0) {{
                reduce_value += (float)a[{idx_exp}];
            }}
        }}
        out[i] = ({type_name})reduce_value;
    }}
}}
        "
        );

        let mut pb = ProgramBuilder::new();
        pb.src(code);
        let src_str = pb.get_src_strings().map_err(|e| e.to_string()).unwrap();
        let cmplr_opts = pb
            .get_compiler_options()
            .map_err(|e| e.to_string())
            .unwrap();
        let program =
            Program::with_source(&context, &src_str, Some(&[device]), &cmplr_opts).unwrap();
        let dims = SpatialDims::One(1);

        let proque = ProQue::new(context.clone(), queue.clone(), program, Some(dims));

        Self {
            proque,
            queue,
            kernel: None,
            dim,
            shape,
            dyn_symbols,
            _phantom: Default::default(),
            dyn_map,
        }
    }
}

impl<T: OclPrm> OpenCLKernel<T> for OpenCLSumReduce<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<Expression> {
        let mut sh = input_shapes[0];
        sh.remove_dim(self.dim);
        vec![sh.n_elements() * size_of::<T>()]
    }
    fn opencl_forward(
        &mut self,
        inputs: &[(&Buffer<T>, ShapeTracker)],
        _: &[&Buffer<T>],
        output_buffers: &[&Buffer<T>],
    ) {
        let mut sh = inputs[0].1;
        sh.remove_dim(self.dim);
        let inp_size = sh.n_elements().to_usize().unwrap();
        let front_size: usize = inputs[0]
            .1
            .dims()
            .iter()
            .take(self.dim)
            .map(|i| i.to_usize().unwrap())
            .product();
        let back_size: usize = inputs[0]
            .1
            .dims()
            .iter()
            .skip(self.dim + 1)
            .map(|i| i.to_usize().unwrap())
            .product();
        let dim_size = inputs[0].1.dims()[self.dim].to_usize().unwrap();

        let mut builder = self.proque.kernel_builder("clkernel");
        builder
            .arg(inputs[0].0)
            .arg(output_buffers[0])
            .arg(inp_size)
            .arg(front_size)
            .arg(back_size)
            .arg(dim_size);

        unsafe {
            for s in &self.dyn_symbols {
                let val = self.dyn_map.as_ref().unwrap()[s] as u32;
                builder.arg(val);
            }
        }

        let kernel = builder.build().unwrap();

        unsafe {
            kernel
                .cmd()
                .global_work_size(inp_size) // real size
                .enq()
                .unwrap();
        }

        self.kernel = Some(Arc::new(kernel));
    }
}

impl<T: OpenCLFloat + OclPrm> Operator for OpenCLSumReduce<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut sh = tensors[0].1;
        sh.remove_dim(self.dim);
        let inp_size = sh.n_elements().to_usize().unwrap();
        let out = Buffer::<T>::builder()
            .queue(self.queue.clone())
            .len((inp_size * size_of::<T>()) as usize) // usize or u64
            .fill_val(Default::default())
            .build()
            .unwrap();

        self.opencl_forward(
            &[(get_buffer_from_tensor(&tensors[0].0), tensors[0].1)],
            &[],
            &[&out],
        );

        unsafe {
            self.kernel.as_ref().unwrap().enq().unwrap();
        }

        let mut curr_data = vec![T::default(); out.len()];
        out.read(&mut curr_data).enq().unwrap();
        for (i, d) in curr_data.iter().enumerate() {
            let val = (*d).to_f32();
            if val.is_nan() || val.is_infinite() {
                panic!("bad value {val} at index {i}");
            }
        }

        vec![Tensor::new(OpenCLBuffer(out))]
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "opencl" {
            return Some(Box::new(OpenCLKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[derive(Clone)]
pub struct OpenCLMaxReduce<T> {
    proque: ProQue,
    queue: Queue,
    kernel: Option<Arc<Kernel>>,
    pub dim: usize,
    pub shape: ShapeTracker,
    dyn_symbols: Vec<char>,
    _phantom: PhantomData<T>,
    dyn_map: *const FxHashMap<char, usize>,
}
crate::debug_type!(OpenCLMaxReduce);

impl<T> PartialEq for OpenCLMaxReduce<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
    }
}

impl<T: OpenCLFloat> OpenCLMaxReduce<T> {
    pub fn new(
        shape: ShapeTracker,
        dim: usize,
        device: Device,
        context: Context,
        queue: Queue,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape], 6);
        let type_name = T::type_name();
        let code = format!(
            "
__kernel void clkernel(const __global float* a, __global float* out, const int n_elements, const int front_size, const int back_size, const int dim_size{rendered}) {{
    int i = get_global_id(0);
    if (i < n_elements) {{
        int a_ = i / back_size;
        int b_ = i % back_size;
        float reduce_value = -INFINITY;
        for (int c_ = 0; c_ < dim_size; c_++) {{
            uint idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid_exp}) != 0) {{
                int a_idx = {idx_exp};
                reduce_value = max(reduce_value, (float)a[a_idx]);
            }}
        }}
        out[i] = ({type_name})reduce_value;
    }}
}}
                "
        );

        let mut pb = ProgramBuilder::new();
        pb.src(code);
        let src_str = pb.get_src_strings().map_err(|e| e.to_string()).unwrap();
        let cmplr_opts = pb
            .get_compiler_options()
            .map_err(|e| e.to_string())
            .unwrap();
        let program =
            Program::with_source(&context, &src_str, Some(&[device]), &cmplr_opts).unwrap();
        let dims = SpatialDims::One(1);

        let proque = ProQue::new(context.clone(), queue.clone(), program, Some(dims));

        Self {
            proque,
            queue,
            kernel: None,
            dim,
            shape,
            dyn_symbols,
            _phantom: Default::default(),
            dyn_map,
        }
    }
}

impl<T: OclPrm> OpenCLKernel<T> for OpenCLMaxReduce<T> {
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<Expression> {
        let mut sh = input_shapes[0];
        sh.remove_dim(self.dim);
        vec![sh.n_elements() * size_of::<T>()]
    }
    fn opencl_forward(
        &mut self,
        inputs: &[(&Buffer<T>, ShapeTracker)],
        _: &[&Buffer<T>],
        output_buffers: &[&Buffer<T>],
    ) {
        let mut sh = inputs[0].1;
        sh.remove_dim(self.dim);
        let inp_size = sh.contiguous().n_elements().to_usize().unwrap();
        let front_size: usize = inputs[0]
            .1
            .dims()
            .iter()
            .take(self.dim)
            .map(|i| i.to_usize().unwrap())
            .product();
        let back_size: usize = inputs[0]
            .1
            .dims()
            .iter()
            .skip(self.dim + 1)
            .map(|i| i.to_usize().unwrap())
            .product();
        let dim_size = inputs[0].1.dims()[self.dim].to_usize().unwrap();

        let mut builder = self.proque.kernel_builder("clkernel");
        builder
            .arg(inputs[0].0)
            .arg(output_buffers[0])
            .arg(inp_size)
            .arg(front_size)
            .arg(back_size)
            .arg(dim_size)
            .build()
            .unwrap();

        unsafe {
            for s in &self.dyn_symbols {
                let val = self.dyn_map.as_ref().unwrap()[s] as u32;
                builder.arg(val);
            }
        }

        let kernel = builder.build().unwrap();

        unsafe {
            kernel
                .cmd()
                .global_work_size(inp_size) // real size
                .enq()
                .unwrap();
        }

        self.kernel = Some(Arc::new(kernel));
    }
}

impl<T: OpenCLFloat + OclPrm> Operator for OpenCLMaxReduce<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = tensors[0]
            .0
            .borrowed()
            .downcast_ref::<OpenCLBuffer<T>>()
            .unwrap();
        let mut sh = tensors[0].1;
        sh.remove_dim(self.dim);

        let inp_size = sh.n_elements().to_usize().unwrap();
        let out = Buffer::<T>::builder()
            .queue(self.queue.clone())
            .len((inp_size * size_of::<T>()) as usize) // usize or u64
            .fill_val(Default::default())
            .build()
            .unwrap();

        self.opencl_forward(&[(a, tensors[0].1)], &[], &[&out]);

        unsafe {
            self.kernel.as_ref().unwrap().enq().unwrap();
        }

        let mut curr_data = vec![T::default(); out.len()];
        out.read(&mut curr_data).enq().unwrap();
        for (i, d) in curr_data.iter().enumerate() {
            let val = (*d).to_f32();
            if val.is_nan() || val.is_infinite() {
                panic!("bad value {val} at index {i}");
            }
        }

        vec![Tensor::new(OpenCLBuffer(out))]
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "opencl" {
            return Some(Box::new(OpenCLKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[derive(Default, Debug)]
pub struct PrimitiveCompiler<T>(PhantomData<T>);

impl<T: OpenCLFloat + 'static + OclPrm> Compiler for PrimitiveCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let platform = Platform::default();
        let device = Device::first(platform).unwrap();
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()
            .unwrap();
        let queue = Queue::new(&context, device, None).unwrap();

        // Go through the graph and insert copy ops
        // Copy function output to device and input from device
        for function_node in graph
            .node_indices()
            .filter(|n| graph.node_weight(*n).unwrap().as_any().is::<LFunction>())
            .collect::<Vec<_>>()
        {
            if graph
                .edges_directed(function_node, petgraph::Direction::Outgoing)
                .count()
                > 0
            {
                // Copy outputs to device
                let sh = ShapeTracker::new(());
                let copy_node = graph
                    .add_op(OpenCLCopyToDevice::<T>::new(queue.clone()))
                    .input(function_node, 0, sh)
                    .finish();

                // Switch outgoing edges from input to copy_node
                for (edge_id, weight, dest) in graph
                    .edges_directed(function_node, petgraph::Direction::Outgoing)
                    .map(|e| (e.id(), *e.weight(), e.target()))
                    .filter(|(_, _, trg)| *trg != copy_node)
                    .collect::<Vec<_>>()
                {
                    graph.add_edge(copy_node, dest, weight);
                    graph.remove_edge(edge_id);
                }

                if graph.no_delete.remove(&function_node) {
                    graph.no_delete.insert(copy_node);
                }
                if let Some(w) = graph.to_retrieve.remove(&function_node) {
                    graph.to_retrieve.insert(copy_node, w);
                }
            }

            // Insert copy from device for function inputs
            for (source, edge, edge_weight) in graph
                .edges_directed(function_node, petgraph::Direction::Incoming)
                .filter(|e| !e.weight().is_schedule())
                .map(|e| (e.source(), e.id(), *e.weight()))
                .collect::<Vec<_>>()
            {
                let (input_order, output_order, shape) = edge_weight.as_data().unwrap();
                let copy_from_node = graph
                    .add_op(OpenCLCopyFromDevice::<T>::default())
                    .input(source, output_order, shape)
                    .finish();
                graph.add_edge(
                    copy_from_node,
                    function_node,
                    Dependency::Data {
                        input_order,
                        output_order: 0,
                        shape,
                    },
                );
                graph.remove_edge(edge);
            }
        }

        // Copy to_retrieve from device
        for (output_node, (_, output_shape)) in graph
            .to_retrieve
            .iter()
            .map(|(a, b)| (*a, *b))
            // Filter to non-functions
            .filter(|(n, _)| !graph.node_weight(*n).unwrap().as_any().is::<LFunction>())
            .collect::<Vec<_>>()
        {
            if graph
                .node_weight(output_node)
                .unwrap()
                .as_any()
                .is::<OpenCLCopyToDevice<T>>()
            {
                // This output is already a copy to, instead of adding a copy from, let's remap back to the source
                let src = graph
                    .neighbors_directed(output_node, petgraph::Direction::Incoming)
                    .next()
                    .unwrap();
                if graph.no_delete.remove(&output_node) {
                    graph.no_delete.insert(src);
                }
                if let Some(w) = graph.to_retrieve.remove(&output_node) {
                    graph.to_retrieve.insert(src, w);
                }
            } else {
                // Create copy node
                let copy_node = graph
                    .add_op(OpenCLCopyFromDevice::<T>::default())
                    .input(output_node, 0, output_shape)
                    .finish();

                remap(output_node, copy_node, &mut ids, graph);
            }
        }

        // Swap primitive ops
        for id in graph.node_indices().collect::<Vec<_>>() {
            let src_shapes = graph
                .edges_directed(id, petgraph::Direction::Incoming)
                .filter_map(|e| e.weight().as_data())
                .sorted_by_key(|e| e.0)
                .map(|e| e.2)
                .collect::<Vec<_>>();
            let op = graph.graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();

            if is::<Log2>(op) {
                *op_ref = Box::new(OpenCLLog2::<T>::new(
                    src_shapes[0],
                    device.clone(),
                    context.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Exp2>(op) {
                *op_ref = Box::new(OpenCLExp2::<T>::new(
                    src_shapes[0],
                    device.clone(),
                    context.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Sin>(op) {
                *op_ref = Box::new(OpenCLSin::<T>::new(
                    src_shapes[0],
                    device.clone(),
                    context.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Sqrt>(op) {
                *op_ref = Box::new(OpenCLSqrt::<T>::new(
                    src_shapes[0],
                    device.clone(),
                    context.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Recip>(op) {
                *op_ref = Box::new(OpenCLRecip::<T>::new(
                    src_shapes[0],
                    device.clone(),
                    context.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(OpenCLContiguous::<T>::new(
                    src_shapes[0],
                    device.clone(),
                    context.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            } else if let Some(c) = op_ref.as_any().downcast_ref::<Constant>() {
                *op_ref = Box::new(OpenCLConstant::<T>(
                    c.0.clone(),
                    queue.clone(),
                    c.1,
                    Default::default(),
                ));
            } else if is::<Add>(op) {
                *op_ref = Box::new(OpenCLAdd::<T>::new(
                    src_shapes[0],
                    src_shapes[1],
                    device.clone(),
                    context.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Mul>(op) {
                *op_ref = Box::new(OpenCLMul::<T>::new(
                    src_shapes[0],
                    src_shapes[1],
                    device.clone(),
                    context.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<LessThan>(op) {
                *op_ref = Box::new(OpenCLMul::<T>::new(
                    src_shapes[0],
                    src_shapes[1],
                    device.clone(),
                    context.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Mod>(op) {
                *op_ref = Box::new(OpenCLMul::<T>::new(
                    src_shapes[0],
                    src_shapes[1],
                    device.clone(),
                    context.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(OpenCLSumReduce::<T>::new(
                    src_shapes[0],
                    *dim,
                    device.clone(),
                    context.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(OpenCLMaxReduce::<T>::new(
                    src_shapes[0],
                    *dim,
                    device.clone(),
                    context.clone(),
                    queue.clone(),
                    &graph.dyn_map,
                ));
            }
        }
    }
}

#[cfg(test)]
#[test]
fn test_binary_unary() {
    use luminal::{prelude::*, tests::assert_close};

    use crate::OpenCLCompiler;
    let mut cx = Graph::new();
    let a = cx.tensor(5).set(vec![1.0, 2.0, 3.0, 4.0, 5.0]).keep();
    let b = cx.tensor(5).set(vec![1.0, 2.0, 3.0, 4.0, 5.0]).keep();
    let mut c = (a + b).exp2().retrieve();

    cx.execute();
    let c_unopt = c.data();
    c.drop();

    cx.compile(OpenCLCompiler::<f32>::default(), &mut c);
    cx.execute();
    assert_close(&c.data(), &c_unopt);
}

#[test]
fn test_binary_prim() {
    use luminal::{prelude::*, tests::assert_close};

    use crate::OpenCLCompiler;
    let mut cx = Graph::new();
    let a = cx.tensor(5).set(vec![1.0, 2.0, 3.0, 4.0, 5.0]).keep();
    let b = cx.tensor(5).set(vec![1.0, 2.0, 3.0, 4.0, 5.0]).keep();
    let mut c = (a + b).retrieve();

    cx.execute();
    let c_unopt = c.data();
    c.drop();

    cx.compile(OpenCLCompiler::<f32>::default(), &mut c);
    cx.execute();
    assert_close(&c.data(), &c_unopt);
}

#[test]
fn test_constant_prim() {
    use luminal::{prelude::*, tests::assert_exact};

    let mut cx = Graph::new();
    let a = cx.constant('a');
    let mut a = (a * a).retrieve();
    cx.compile(OpenCLCompiler::<f32>::default(), &mut a);

    cx.set_dyn_dim('a', 10);
    cx.execute();
    assert_exact(&a.data(), &[100.0]);
    a.drop();
    cx.set_dyn_dim('a', 25);
    cx.execute();
    assert_exact(&a.data(), &[625.0]);
}

#[test]
fn test_sum_reduce_prim() {
    use luminal::{prelude::*, tests::assert_exact};

    let mut cx = Graph::new();
    let a = cx
        .tensor((2, 2, 3))
        .set([[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]);
    let mut b = a.sum(1).retrieve();

    cx.compile(OpenCLCompiler::<f32>::default(), &mut b);
    cx.execute();
    assert_exact(&b.data(), &[2.0, 4.0, 6.0, 2.0, 4.0, 6.0]);
}

#[test]
fn test_max_reduce_prim() {
    use luminal::{prelude::*, tests::assert_exact};

    let mut cx = Graph::new();
    let a = cx
        .tensor((2, 2, 3))
        .set([[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]);
    let mut b = a.max(1).retrieve();

    cx.compile(OpenCLCompiler::<f32>::default(), &mut b);
    cx.execute();
    assert_exact(&b.data(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}
