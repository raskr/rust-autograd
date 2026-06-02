//! ## Variable and namespace
//!
//! [Tensor] can behave like a trainable variable if the corresponding NdArray were registered in a [VariableEnvironment].
//!
//! ### Basic usages
//!
//! ```
//! use autograd as ag;
//! use ag::ndarray_ext;
//! use ag::variable::{VariableID, NamespaceTrait};
//! use ag::Tensor;
//! use ag::prelude::*;
//!
//! let mut env = ag::VariableEnvironment::new();
//!
//! // Register variable arrays in the *default* namespace.
//! // `set` method returns the id of the given array;
//! let a: VariableID = env.set(ndarray_ext::zeros(&[1, 10]));
//!
//! // You can name arrays and lookup them later
//! let b: VariableID = env.name("b")
//!                        .set(ndarray_ext::zeros(&[1, 10]));
//!
//! // Register variable arrays in the `my_namespace` namespace.
//! let c: VariableID = env.namespace_mut("my_namespace")
//!     .slot()
//!     .name("c")
//!     .set(ndarray_ext::zeros(&[1, 10]));
//!
//! // Create and run some graphs with the env.
//! for epoch in 0..10 {
//!     // use VariableEnvironment::run() to lookup the vars.
//!     env.run(|ctx| {
//!         // Lookup variable tensors.
//!         let _: Tensor<f32> = ctx.variable(a); // with VariableID
//!         let _: Tensor<f32> = ctx.variable("b"); // with name in the default namespace
//!         let _: Tensor<f32> = ctx.variable(("my_namespace", "c")); // with namespace/name
//!
//!         // Access ns through the context
//!         let ns = ctx.namespace("my_namespace");
//!     })
//! }
//!
//! // Collecting var names in a specific namespace.
//! let names_: Vec<&str> = env.default_namespace().current_var_names();
//! let names_: Vec<&str> = env.namespace("my_namespace").current_var_names();
//! ```
//!
//! See also neural network examples in `examples` directory.
//!
//! # Model persistence
//! ```
//! use autograd as ag;
//! use std::fs;
//! use std::error::Error;
//!
//! let dir = "/tmp/rust-autograd/test/model_persistence";
//! fs::create_dir_all(dir).unwrap();
//! let path = format!("{}/model.json", dir);
//! let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
//!
//! let mut env = ag::VariableEnvironment::new();
//! env.slot().name("a").set(rng.standard_normal(&[2, 3]));
//! env.slot().name("b").set(rng.standard_normal(&[2, 3]));
//!
//! // save
//! env.save(&path).unwrap();
//!
//! // load it
//! let loaded_env = ag::VariableEnvironment::<f64>::load(&path).unwrap();
//!
//! // alternatively, it's possible to initialize the existing env
//! let mut new_env = ag::VariableEnvironment::<f64>::new();
//! let _: Result<(), Box<dyn Error>> = new_env.initialize(path);
//!
//! // new_env.run(...
//! ```
use crate::graph::Context;
use crate::{uuid::Uuid, Float, FxHashMap, Graph, NdArray, NdArrayView, NdArrayViewMut, Tensor};
use serde::Deserialize;
use serde_json;
use smallvec::alloc::fmt::Formatter;
use std::cell::RefCell;

use std::error::Error;
use std::fs::File;
use std::ops::Deref;
use std::path::Path;

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Serialize, Deserialize)]
/// Variable array's ID that is unique in a `VariableEnvironment`.
///
/// See [`VariableEnvironment`].
pub struct VariableID(pub(crate) usize);

impl From<usize> for VariableID {
    fn from(a: usize) -> VariableID {
        VariableID(a)
    }
}

impl From<VariableID> for usize {
    fn from(a: VariableID) -> usize {
        a.0
    }
}

impl std::fmt::Display for VariableID {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

const DEFAULT_NAMESPACE_ID: &'static str = "";

pub type Variable<F> = RefCell<NdArray<F>>;

/// Get or create a variable tensor.
pub trait GetVariableTensor<'g, F: Float, Arg> {
    fn variable(&'g self, id: Arg) -> Tensor<'g, F>;
}

impl<'g, 'e: 'g, F: Float> GetVariableTensor<'g, F, &'static str> for Context<'e, F> {
    /// Get or create a variable tensor by name in the default namespace.
    fn variable(&'g self, name: &str) -> Tensor<'g, F> {
        self.graph
            .variable_by_name(name, &self.var_env_ref.default_namespace())
    }
}

impl<'g, 'e: 'g, F: Float> GetVariableTensor<'g, F, VariableID> for Context<'e, F> {
    /// Get or create a variable tensor by [`VariableID`]
    fn variable(&'g self, id: VariableID) -> Tensor<'g, F> {
        self.graph.variable_by_id(id)
    }
}

impl<'g, 'e: 'g, F: Float> GetVariableTensor<'g, F, (&'static str, &'static str)>
    for Context<'e, F>
{
    /// Get or create a variable tensor by VariableID
    fn variable(&'g self, id: (&'static str, &'static str)) -> Tensor<'g, F> {
        self.graph
            .variable_by_name(id.1, &self.var_env_ref.namespace(id.0))
    }
}

/// Manages variable arrays
///
/// See [variable](crate::variable).
#[derive(Clone)]
pub struct VariableEnvironment<F> {
    pub(crate) array_list: Vec<Variable<F>>,
    pub(crate) name_to_id: FxHashMap<FullName, VariableID>,
}

// Identifies variable array
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct FullName {
    pub(crate) namespace_id: String,
    pub(crate) variable_name: String,
}

/// Anonymous slot to register a variable
///
/// The registered variable array will be kept in the associated namespace.
///
/// Use `VariableNamespaceMut::slot` to instantiate this.
pub struct VariableSlot<'ns, 'env, F: Float> {
    namespace: &'ns mut VariableNamespaceMut<'env, F>,
}

/// Named slot to register a variable
///
/// Returned by `VariableSlot::name` etc.
///
/// The registered variable array will be kept in the associated namespace.
/// You can lookup the array's tensor representation using the name later.
pub struct NamedVariableSlot<'ns, 'env, F: Float, S: Into<String>> {
    namespace: &'ns mut VariableNamespaceMut<'env, F>,
    name: S,
}

/// Anonymous slot to register a variable
///
/// The registered variable array will be kept in the *default* namespace.
pub struct DefaultVariableSlot<'env, F: Float> {
    env: &'env mut VariableEnvironment<F>,
}

/// Named slot where a variable array can be registered
///
/// The registered variable array will be kept in the *default* namespace.
/// You can lookup the array's tensor representation using the name later.
pub struct NamedDefaultVariableSlot<'env, F: Float, S: Into<String>> {
    env: &'env mut VariableEnvironment<F>,
    name: S,
}

/// Manages variable arrays using their unique names.
///
/// Each of the variables managed by autograd is always associated to a single namespace.
/// See [variable](crate::variable).
pub struct VariableNamespace<'env, F: Float> {
    pub(crate) env: &'env VariableEnvironment<F>,
    pub(crate) namespace_id: &'static str,
}

/// Mutable version of `VariableNamespace`.
///
/// You can register a new variable array with this namespace using `slot` method.
pub struct VariableNamespaceMut<'env, F: Float> {
    pub(crate) env: &'env mut VariableEnvironment<F>,
    pub(crate) namespace_id: &'static str,
}

impl FullName {
    fn new(namespace_id: &'static str, variable_name: String) -> Self {
        FullName {
            namespace_id: namespace_id.to_string(),
            variable_name,
        }
    }

    fn to_string(&self) -> String {
        let ns = self.namespace_id.deref();
        let name = self.variable_name.deref();
        format!("{}\u{00001}{}", ns, name)
    }
}

pub trait NamespaceTrait<F: Float> {
    /// The name of this namespace
    fn name(&self) -> &'static str;

    /// A reference to the `VariableEnvironment`.
    fn env(&self) -> &VariableEnvironment<F>;

    /// Returns a reference to the variable array
    #[inline]
    fn get_array_by_id(&self, vid: VariableID) -> &RefCell<NdArray<F>> {
        &self.env().array_list[vid.0]
    }

    /// Returns a reference to the variable array with the specified name.
    ///
    /// Returns `None` if the given name is not valid in this namespace.
    #[inline]
    fn get_array_by_name<S: AsRef<str>>(&self, name: S) -> Option<&RefCell<NdArray<F>>> {
        let name = &FullName::new(self.name(), name.as_ref().to_string());
        self.env()
            .name_to_id
            .get(name)
            .map(|vid| &self.env().array_list[vid.0])
    }

    /// Lists all the IDs of the variable arrays in this namespace.
    fn current_var_ids(&self) -> Vec<VariableID> {
        self.env()
            .name_to_id
            .iter()
            .filter_map(|(v_name, &vid)| {
                if v_name.namespace_id == self.name() {
                    Some(vid)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Lists all the names of the variable arrays in this namespace.
    fn current_var_names(&self) -> Vec<&str> {
        self.env()
            .name_to_id
            .iter()
            .filter_map(|(v_name, _)| {
                if v_name.namespace_id == self.name() {
                    Some(v_name.variable_name.deref())
                } else {
                    None
                }
            })
            .collect()
    }
}

impl<'ns, 'env, F: Float, S: Into<String>> NamedVariableSlot<'ns, 'env, F, S> {
    /// Registers the given name and array with the specified namespace.
    pub fn set<D: ndarray::Dimension>(self, v: ndarray::Array<F, D>) -> VariableID {
        register_variable(
            v,
            self.namespace.namespace_id,
            self.name.into(),
            self.namespace.env,
        )
    }
}

impl<'env, F: Float> DefaultVariableSlot<'env, F> {
    /// Registers the given array with the *default* namespace.
    pub fn set<D: ndarray::Dimension>(self, v: ndarray::Array<F, D>) -> VariableID {
        register_variable(
            v,
            DEFAULT_NAMESPACE_ID,
            Uuid::new_v4().to_string(),
            self.env,
        )
    }

    /// Specifies the name for the array that will be registered.
    pub fn name<S: Into<String>>(self, name: S) -> NamedDefaultVariableSlot<'env, F, S> {
        NamedDefaultVariableSlot {
            env: self.env,
            name,
        }
    }
}

impl<'env, F: Float, S: Into<String>> NamedDefaultVariableSlot<'env, F, S> {
    /// Registers the given name and array with the specified namespace.
    pub fn set<D: ndarray::Dimension>(self, v: ndarray::Array<F, D>) -> VariableID {
        register_variable(v, DEFAULT_NAMESPACE_ID, self.name.into(), self.env)
    }
}

impl<'ns, 'env, F: Float> VariableSlot<'ns, 'env, F> {
    /// Registers the given array with the specified namespace.
    pub fn set<D: ndarray::Dimension>(self, v: ndarray::Array<F, D>) -> VariableID {
        register_variable(
            v,
            self.namespace.namespace_id,
            Uuid::new_v4().to_string(),
            self.namespace.env,
        )
    }

    /// Specifies the name for the array that will be registered.
    pub fn name<S: Into<String>>(self, name: S) -> NamedVariableSlot<'ns, 'env, F, S> {
        NamedVariableSlot {
            namespace: self.namespace,
            name,
        }
    }
}

fn register_variable<F: Float, D: ndarray::Dimension, S: Into<String>>(
    v: ndarray::Array<F, D>,
    namespace_id: &'static str,
    variable_name: S,
    env: &mut VariableEnvironment<F>,
) -> VariableID {
    let vid = FullName::new(namespace_id, variable_name.into());
    let next_id = env.array_list.len().into();
    env.name_to_id.insert(vid, next_id);
    env.array_list.push(RefCell::new(v.into_dyn()));
    next_id
}

impl<'env, F: Float> NamespaceTrait<F> for VariableNamespace<'env, F> {
    #[inline]
    fn name(&self) -> &'static str {
        self.namespace_id
    }
    #[inline]
    fn env(&self) -> &VariableEnvironment<F> {
        self.env
    }
}

impl<'env, F: Float> NamespaceTrait<F> for VariableNamespaceMut<'env, F> {
    #[inline]
    fn name(&self) -> &'static str {
        self.namespace_id
    }
    #[inline]
    fn env(&self) -> &VariableEnvironment<F> {
        self.env
    }
}

impl<'e, F: Float> VariableNamespace<'e, F> {
    /// Returns an iterator of variable arrays and their names in this namespace
    #[allow(unused)]
    pub fn iter(&self) -> impl Iterator<Item = (&str, &RefCell<NdArray<F>>)> {
        iter(self)
    }
}

impl<'e, F: Float> VariableNamespaceMut<'e, F> {
    /// Returns an iterator of variable arrays and their names in this namespace
    #[allow(unused)]
    pub fn iter(&self) -> impl Iterator<Item = (&str, &RefCell<NdArray<F>>)> {
        iter(self)
    }
}

fn iter<F: Float>(
    ns: &impl NamespaceTrait<F>,
) -> impl Iterator<Item = (&str, &RefCell<NdArray<F>>)> {
    ns.env().name_to_id.iter().filter_map(move |ent| {
        // filter out other namespaces
        if &ent.0.namespace_id == ns.name() {
            Some((
                ent.0.variable_name.deref(),
                ns.get_array_by_name(ent.0.variable_name.deref()).unwrap(),
            ))
        } else {
            None
        }
    })
}
impl<'ns, 'env, F: Float> VariableNamespaceMut<'env, F> {
    /// Makes a temporary slot for registering a variable array in this namespace.
    pub fn slot(&'ns mut self) -> VariableSlot<'ns, 'env, F> {
        VariableSlot { namespace: self }
    }
}

#[test]
fn test_env_iter() {
    use crate::ndarray_ext;

    let mut env = VariableEnvironment::<f32>::new();
    let v1 = env.slot().set(ndarray_ext::zeros(&[3, 2]));
    let v2 = env.slot().set(ndarray_ext::zeros(&[2, 3]));
    for (i, (vid, arr)) in env.iter().enumerate() {
        if i == 0 {
            assert_eq!(vid, v1);
            assert_eq!(arr.borrow().shape(), &[3, 2]);
        }
        if i == 1 {
            assert_eq!(vid, v2);
            assert_eq!(arr.borrow().shape(), &[2, 3]);
        }
    }
}

use crate::ndarray_ext;

#[test]
fn test_namespace_iter() {
    let mut env = VariableEnvironment::<f32>::new();
    env.slot().name("v1").set(ndarray_ext::zeros(&[3, 2]));
    env.slot().name("v2").set(ndarray_ext::zeros(&[2, 3]));

    for (i, (name, arr)) in env.default_namespace().iter().enumerate() {
        if i == 0 {
            assert_eq!(name, "v1");
            assert_eq!(arr.borrow().shape(), &[3, 2]);
        }
        if i == 1 {
            assert_eq!(name, "v2");
            assert_eq!(arr.borrow().shape(), &[2, 3]);
        }
    }

    for (i, (name, arr)) in env.default_namespace_mut().iter().enumerate() {
        if i == 0 {
            assert_eq!(name, "v1");
            assert_eq!(arr.borrow().shape(), &[3, 2]);
        }
        if i == 1 {
            assert_eq!(name, "v2");
            assert_eq!(arr.borrow().shape(), &[2, 3]);
        }
    }
}

#[derive(Serialize)]
struct SerializableVariableEnvironment<'a, F> {
    array_list: &'a Vec<Variable<F>>,
    name_to_id: FxHashMap<String, VariableID>,
}

#[derive(Deserialize)]
struct DeserializedVariableEnvironment<F> {
    array_list: Vec<Variable<F>>,
    name_to_id: FxHashMap<String, VariableID>,
}

// f32 save and load
impl<'env> VariableEnvironment<f32> {
    /// Creates a new `VariableEnvironment` using the one that was previously persisted.
    ///
    /// Returns the result of the execution.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<VariableEnvironment<f32>, Box<dyn Error>> {
        let raw: DeserializedVariableEnvironment<f32> = Self::deserialize(path)?;
        Self::load_internal(raw)
    }

    /// Initialize this instance with the one that was previously persisted.
    pub fn initialize<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn Error>> {
        let raw: DeserializedVariableEnvironment<f32> = Self::deserialize(path)?;
        let VariableEnvironment {
            array_list,
            name_to_id,
        } = Self::load_internal(raw)?;
        self.array_list = array_list;
        self.name_to_id = name_to_id;
        Ok(())
    }
}

// f64 save and load
impl<'env> VariableEnvironment<f64> {
    /// Creates a new `VariableEnvironment` using the one that was previously persisted.
    ///
    /// Returns the result of the execution.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<VariableEnvironment<f64>, Box<dyn Error>> {
        let raw: DeserializedVariableEnvironment<f64> = Self::deserialize(path)?;
        Self::load_internal(raw)
    }

    /// Initialize this instance with the one that was previously persisted.
    pub fn initialize<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn Error>> {
        let raw: DeserializedVariableEnvironment<f64> = Self::deserialize(path)?;
        let VariableEnvironment {
            array_list,
            name_to_id,
        } = Self::load_internal(raw)?;
        self.array_list = array_list;
        self.name_to_id = name_to_id;
        Ok(())
    }
}

impl<'env, F: Float> VariableEnvironment<F> {
    // New
    pub fn new() -> VariableEnvironment<F> {
        Self {
            name_to_id: FxHashMap::default(),
            array_list: Vec::new(),
        }
    }

    /// Returns an iterator of the variable arrays and their ids in this env.
    #[allow(unused)]
    pub fn iter(&'env self) -> impl Iterator<Item = (VariableID, &RefCell<NdArray<F>>)> {
        self.array_list
            .iter()
            .enumerate()
            .map(|(i, v)| (VariableID::from(i), v))
    }

    /// Saves the current VariableEnvironment to storage.
    ///
    /// Returns the result of the execution.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        let f = File::create(path.as_ref())?;
        serde_json::to_writer(f, &self.prepare_for_serde())?;
        Ok(())
    }

    fn deserialize<T, P: AsRef<Path>>(path: P) -> Result<T, Box<dyn Error>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let f = File::open(path.as_ref())?;
        let ret = serde_json::from_reader(f)?;
        Ok(ret)
    }

    fn load_internal<T>(
        env: DeserializedVariableEnvironment<T>,
    ) -> Result<VariableEnvironment<T>, Box<dyn Error>> {
        let name_to_id: FxHashMap<FullName, VariableID> = env
            .name_to_id
            .iter()
            .map(|(fullname, &vid)| {
                let mut split = fullname.split("\u{0001}").into_iter();
                let namespace_id = split.next().unwrap().to_owned();
                let var_name = split.next().unwrap().to_owned();
                let fullname = FullName {
                    namespace_id,
                    variable_name: var_name,
                };
                (fullname, vid)
            })
            .collect();

        Ok(VariableEnvironment {
            array_list: env.array_list,
            name_to_id,
        })
    }

    fn prepare_for_serde(&self) -> SerializableVariableEnvironment<F> {
        let name_to_id: FxHashMap<String, VariableID> = self
            .name_to_id
            .iter()
            .map(|(fullname, vid)| (fullname.to_string(), *vid))
            .collect();
        SerializableVariableEnvironment {
            array_list: &self.array_list,
            name_to_id,
        }
    }

    /// Makes a temporary slot for registering a variable array in the *default* namespace.
    pub fn slot(&'env mut self) -> DefaultVariableSlot<'env, F> {
        DefaultVariableSlot { env: self }
    }

    /// Registers the given array with the *default* namespace.
    pub fn set<D: ndarray::Dimension>(&'env mut self, v: ndarray::Array<F, D>) -> VariableID {
        register_variable(v, DEFAULT_NAMESPACE_ID, Uuid::new_v4().to_string(), self)
    }

    /// Prepares a slot for the *default* namespace to register a variable array
    pub fn name<S: Into<String>>(&'env mut self, name: S) -> NamedDefaultVariableSlot<'env, F, S> {
        NamedDefaultVariableSlot { env: self, name }
    }

    /// Get or create a namespace with specified id.
    ///
    /// See [variable](crate::variable).
    /// Same as [`Context::namespace`](Context::namespace()).
    #[inline]
    pub fn namespace(&'env self, namespace_id: &'static str) -> VariableNamespace<'env, F> {
        VariableNamespace {
            namespace_id,
            env: self,
        }
    }

    /// Get or create a mutable namespace with specified name.
    ///
    /// Return value is used for variable registration.
    /// See [variable](crate::variable).
    #[inline]
    pub fn namespace_mut(
        &'env mut self,
        namespace_id: &'static str,
    ) -> VariableNamespaceMut<'env, F> {
        VariableNamespaceMut {
            namespace_id,
            env: self,
        }
    }

    /// Get or create the *default* namespace.
    ///
    /// See [variable](crate::variable).
    /// Same as [`Context::default_namespace`](Context::default_namespace).
    #[inline]
    pub fn default_namespace(&'env self) -> VariableNamespace<'env, F> {
        self.namespace(DEFAULT_NAMESPACE_ID)
    }

    /// Get or create a mutable *default* namespace.
    ///
    /// Return value is used for variable registration.
    #[inline]
    pub fn default_namespace_mut(&'env mut self) -> VariableNamespaceMut<'env, F> {
        self.namespace_mut(DEFAULT_NAMESPACE_ID)
    }

    /// Returns a reference to the variable array with the specified id.
    ///
    /// `VariableID` is returned by the `*Slot::set`.
    #[inline]
    pub fn get_array_by_id(&self, vid: VariableID) -> Option<&RefCell<NdArray<F>>> {
        self.array_list.get(vid.0)
    }

    /// Creates a computation graph associated with this `VariableEnvironment`.
    ///
    /// See [variable](crate::variable).
    pub fn run<FN, R>(&'env self, f: FN) -> R
    where
        FN: FnOnce(&mut Context<'env, F>) -> R,
    {
        let g = Graph {
            node_set: RefCell::new(Vec::with_capacity(256)),
            variable2node: RefCell::new(FxHashMap::default()),
        };
        let mut c = Context {
            var_env_ref: self,
            graph: g,
        };
        f(&mut c)
    }

    pub(crate) fn as_view(&self, vid: VariableID) -> NdArrayView<F> {
        unsafe {
            self.array_list[vid.0]
                .borrow()
                .raw_view()
                .clone()
                .deref_into_view()
        }
    }

    pub(crate) fn as_view_mut(&self, vid: VariableID) -> NdArrayViewMut<F> {
        unsafe {
            self.array_list[vid.0]
                .borrow_mut()
                .raw_view_mut()
                .clone()
                .deref_into_view_mut()
        }
    }
}

impl<'t, 'g, F: Float> Graph<F> {
    /// Same as `Context::variable(VariableID)`
    #[inline]
    pub fn variable_by_id(&'g self, vid: VariableID) -> Tensor<'g, F> {
        let tid = {
            let temp = self.variable2node.borrow();
            temp.get(&vid).cloned()
        };
        if let Some(tid) = tid {
            // use existing tensor
            self.tensor(tid)
        } else {
            // allocate a new tensor
            let allocated = Tensor::builder(self)
                .set_variable(vid)
                .build(crate::tensor_ops::basic_source_ops::Variable);
            // register vid -> tid map
            self.variable2node.borrow_mut().insert(vid, allocated.id);
            allocated
        }
    }

    /// Same as `Context::variable((namespace, name))`
    pub fn variable_by_name<S: AsRef<str>>(
        &self,
        name: S,
        namespace: &impl NamespaceTrait<F>,
    ) -> Tensor<F> {
        let full_name = &FullName::new(namespace.name(), name.as_ref().to_string());
        if let Some(&vid) = namespace.env().name_to_id.get(full_name) {
            // find VariableID
            self.variable_by_id(vid)
        } else {
            let ns = namespace.name();
            if ns.len() == 0 {
                panic!(
                    "variable array not found in default namespace: {}",
                    name.as_ref()
                )
            } else {
                panic!(
                    "variable array `{}` not found in namespace {}",
                    name.as_ref(),
                    ns
                )
            }
        }
    }

    /// Get tensors with their variable ids.
    ///
    /// See `VariableEnvironment` for the usages.
    pub fn var_tensors_by_id<'e: 'g>(
        &'g self,
        env: &'e VariableEnvironment<F>,
    ) -> impl Iterator<Item = (VariableID, Tensor<'g, F>)> {
        (0..env.array_list.len()).map(move |vid| (vid.into(), self.variable_by_id(vid.into())))
    }

    /// Get tensors and their variable names in the specified namespace.
    ///
    /// See `VariableEnvironment` for the usages.
    pub fn var_tensors_by_name<'ns, 'e: 'g>(
        &'g self,
        ns: &'ns VariableNamespace<'e, F>,
    ) -> impl Iterator<Item = (&'ns str, Tensor<'g, F>)> {
        ns.env().name_to_id.iter().filter_map(move |ent| {
            // filter out other namespaces
            if &ent.0.namespace_id == ns.name() {
                Some((ent.0.variable_name.deref(), self.variable_by_id(*ent.1)))
            } else {
                None
            }
        })
    }
}

#[allow(unused)]
fn compile_common_usages() {
    use crate::prelude::*;
    use crate::tensor_ops as T;

    let mut env = VariableEnvironment::<f32>::new();
    // let _cur_names_ = env.default_namespace().current_var_names();

    env.run(|g| {
        let ns = g.env().default_namespace();

        let _v3_ = g.variable_by_name("a", &ns);
        let v = g.variable("a");
        let v2 = g.variable(VariableID(0));
        let v3 = g.variable(("my_ns", "a"));
        let ones = T::zeros(&[1], g) + v + v2 + v3;
        let _ = ones.eval(g);
    });

    env.run(|g| {
        let ns = g.env().default_namespace();
        let v = g.variable("a");
        let _ = v.eval(g);
    })
}

#[test]
fn save_and_load() {
    use approx::AbsDiffEq;
    use std::collections::HashMap;
    use std::fs;

    let dir = "/tmp/rust-autograd/test/save_and_load";
    fs::create_dir_all(dir).unwrap();
    let path = format!("{}/model.json", dir);
    let rng = ndarray_ext::ArrayRng::<f64>::default();

    let mut env = VariableEnvironment::new();
    env.slot().name("a").set(rng.standard_normal(&[2, 3]));
    env.slot().name("b").set(rng.standard_normal(&[2, 3]));

    // save
    env.save(&path).unwrap();

    // load and assert
    {
        let loaded_env = VariableEnvironment::<f64>::load(&path).unwrap();

        // assert array equalities
        for (vid, array) in env.iter() {
            let loaded_env_map: HashMap<_, _> = loaded_env.iter().collect();
            let loaded_array = loaded_env_map.get(&vid).unwrap();
            assert!(array.abs_diff_eq(*loaded_array, 1e-6));
        }

        assert_eq!(env.name_to_id, loaded_env.name_to_id);
    }
}

#[test]
fn save_and_init() {
    use std::fs;

    let dir = "/tmp/rust-autograd/test/save_and_init";
    fs::create_dir_all(dir).unwrap();
    let path = format!("{}/model.json", dir);
    let rng = ndarray_ext::ArrayRng::<f64>::default();

    let mut env = VariableEnvironment::new();
    let a = env.name("a").set(rng.standard_normal(&[2, 3]));
    let b = env.name("b").set(rng.standard_normal(&[2, 3]));

    for _ in 0..10 {
        env.run(|g| {
            let _a_ = g.variable(a);
            let _b_ = g.variable(b);
            g.env().save(&path).unwrap();
        });
    }

    env.initialize(&path).unwrap();
}
