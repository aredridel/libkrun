//! JSON schema for ffier library metadata.
//!
//! This is the universal binding description format. A program in any language
//! can read this JSON and generate complete bindings to the Rust library —
//! including doc comments, lifetime relationships, C type names, and error
//! variant codes.
//!
//! All types are registered in a single `type_registry` keyed by Rust type
//! name. Params and returns reference types by name + usage modifiers
//! (ref kind, lifetimes).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Top-level library
// ---------------------------------------------------------------------------

/// Complete description of an ffier library.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Library {
    /// FFI prefix (e.g. "ft" → functions are `ft_widget_new`, C types are `FtWidget`).
    pub prefix: String,
    /// Override prefix for primitive types (Str, Bytes, Result, VtableHandle).
    /// When set, these use a different prefix from the library's own
    /// (e.g. `"krun"` for a library with prefix `"krun_init"`).
    /// When `None`, defaults to `prefix`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primitives_prefix: Option<String>,
    /// All types used in the library, keyed by Rust name.
    /// Includes primitives (`"i32"`), builtins (`"str"`), handles (`"Widget"`),
    /// errors (`"TestError"`), and traits (`"Fruit"`).
    pub type_registry: BTreeMap<String, TypeEntry>,
    pub exported_types: Vec<ExportedType>,
    pub errors: Vec<ErrorType>,
    /// Plain enums exported as C `#define` constants.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub enum_constants: Vec<EnumType>,
    /// Bitflags types exported as C `#define` constants.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bitflags_constants: Vec<EnumType>,
    /// Free (non-method) functions exported via `#[ffier::export]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub free_functions: Vec<FreeFunction>,
    pub traits: Vec<ImplementableTrait>,
    pub trait_impls: Vec<TraitImpl>,
}

// ---------------------------------------------------------------------------
// Type registry
// ---------------------------------------------------------------------------

/// Semantic tags that inform generators how to handle a type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Blessing {
    /// The error dispatch trait (provides `code`, `message`, `result`).
    ErrorTrait,
    /// FFI result type (e.g. `uint64_t` typedef).
    Result,
    /// UTF-8 string (ptr + len struct).
    Str,
    /// Byte slice (ptr + len struct).
    Bytes,
    /// Filesystem path (ptr + len, not necessarily UTF-8).
    Path,
    /// Stack-allocated vtable handle for passing trait objects.
    VtableHandle,
    /// Raw file descriptor (platform-level integer).
    RawFd,
    /// Borrowed file descriptor (non-owning).
    BorrowedFd,
    /// Owned file descriptor (transfers ownership).
    OwnedFd,
    /// Builder method return — `void` at C level, `-> Self` in Rust.
    ReplacesSelf,
    /// Streaming string writer trait (PushStr).
    PushStr,
    /// Object array (contiguous borrowed-handle array + len struct).
    ObjectArray,
    /// Generic handle type (any concrete handle).
    Object,
}

/// An entry in the type registry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeEntry {
    /// What kind of type this is.
    pub kind: TypeKind,
    /// Stable type tag from `library_definition!`. Only present for
    /// handles, errors, and exported traits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub type_tag: Option<u32>,
    /// Optional blessing for well-known types.
    /// Generators use this to locate special types without hardcoding names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bless: Option<Blessing>,
    /// Lifetime parameters on the type definition (e.g. `["a"]` for `View<'a>`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lifetime_params: Vec<String>,
}

/// The kind of a type in the registry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypeKind {
    /// Primitive integer or bool: `i32`, `u64`, `bool`, `usize`, etc.
    Primitive {
        /// C type name (e.g. `"int32_t"`, `"uint64_t"`, `"bool"`).
        c_type: std::string::String,
    },
    /// String slice: `str`. Passed as a struct (ptr+len).
    String {
        /// Library-prefixed C struct name (e.g. `"FtStr"`).
        c_name: std::string::String,
    },
    /// Byte slice: `[u8]`. Passed as a struct (ptr+len).
    Bytes {
        /// Library-prefixed C struct name (e.g. `"FtBytes"`).
        c_name: std::string::String,
    },
    /// Type alias for another type (e.g. `BorrowedFd` → `RawFd`).
    Alias {
        /// The underlying type name this aliases.
        alias_of: std::string::String,
    },
    /// Opaque handle type (a struct exported via `#[ffier::export]`).
    Handle {
        /// Library-prefixed C typedef name (e.g. `"FtWidget"`).
        c_name: std::string::String,
    },
    /// Error type (an enum exported via `#[derive(FfiError)]`).
    Error {
        /// Library-prefixed C typedef name (e.g. `"FtTestError"`).
        c_name: std::string::String,
    },
    /// Trait (from `#[ffier::export]` on a trait definition, or discovered via trait impls).
    Trait {
        /// Library-prefixed C typedef name (e.g. `"FtFruit"`).
        c_name: std::string::String,
    },
    /// Plain enum with explicit discriminant values and a `#[repr(uN)]`.
    /// At the C ABI level the parameter is the underlying integer type
    /// (resolved via `alias_of`), but the schema carries the variant names
    /// and values so generators can emit `#define` constants / typed enums.
    Enum {
        /// The underlying integer type name (e.g. `"u32"`, `"u64"`).
        alias_of: std::string::String,
    },
    /// Bitflags type — a newtype struct over an integer with named flag
    /// constants. Like `Enum`, the C ABI passes the underlying integer
    /// type by value. Generators emit `#define` constants for C and
    /// `bitflags!` invocations for Rust client code.
    Bitflags {
        /// The underlying integer type name (e.g. `"u32"`, `"u64"`).
        alias_of: std::string::String,
    },
    /// Handle type from a foreign ffier library. Passed as `*mut c_void`
    /// across the C ABI but the type definition lives in the foreign
    /// library's generated client crate, not in this library's.
    /// Generators should not emit a struct definition for this type;
    /// instead they should reference it from the foreign client crate.
    ForeignHandle {
        /// The Rust crate path that provides this type (e.g.
        /// `"ffier_test_foreign_lib_via_cdylib"`). Generators emit
        /// a `use` import for this crate and reference the type directly.
        foreign_crate: String,
        /// C typedef name (e.g. `"FlForeignConfig"`). Used by the C header
        /// generator to emit typed parameters instead of `void*`.
        c_name: String,
    },
}

impl TypeKind {
    /// True for `Handle` and `ForeignHandle` — any opaque handle type.
    pub fn is_handle(&self) -> bool {
        matches!(
            self,
            TypeKind::Handle { .. } | TypeKind::ForeignHandle { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// Type references (used in params, returns)
// ---------------------------------------------------------------------------

/// A reference to a type in the registry, with usage-site modifiers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeRef {
    /// Key into `type_registry` (e.g. `"Widget"`, `"i32"`, `"str"`).
    #[serde(rename = "type")]
    pub type_name: String,
    /// How the type is accessed at this usage site.
    #[serde(default, skip_serializing_if = "is_ref_none")]
    pub ref_kind: RefKind,
    /// Lifetime on the reference itself (e.g. `"a"` in `&'a Widget`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ref_lifetime: Option<String>,
    /// Lifetime arguments applied to the type's params
    /// (e.g. `["a"]` in `View<'a>`, `["b"]` in `&'a View<'b>`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub type_args: Vec<String>,
    /// Whether this type is wrapped in `Option<_>`. At the C ABI level
    /// the representation is the same (e.g. `FfierBytes` with null data
    /// for `None`), but generators use this to emit `Option<&str>` in
    /// Rust client code.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub optional: bool,
    /// Whether this type is owned via `Box<_>`. At the C ABI level the
    /// representation is the same (e.g. `FfierBytes`), but generators
    /// use this to emit `Box<str>` and the caller must free the value.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub owned: bool,
}

/// How a type is accessed at a usage site.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefKind {
    /// By value.
    #[default]
    None,
    /// `&T` or `&'a T`.
    Shared,
    /// `&mut T` or `&'a mut T`.
    Mut,
}

impl TypeRef {
    /// Reconstruct the Rust type syntax from this reference.
    /// E.g. `TypeRef { type: "Widget", ref_kind: Shared, ref_lifetime: Some("a") }` → `&'a Widget`
    /// E.g. `TypeRef { type: "View", type_args: ["a"] }` → `View<'a>`
    pub fn to_rust_type(&self) -> String {
        let mut s = self.to_rust_type_inner(false);
        if self.owned {
            s = format!("Box<{s}>");
        }
        if self.optional {
            s = format!("Option<{s}>");
        }
        s
    }

    fn to_rust_type_inner(&self, use_static: bool) -> String {
        let mut s = String::new();

        // Reference prefix
        match self.ref_kind {
            RefKind::None => {}
            RefKind::Shared => {
                s.push('&');
                if use_static {
                    s.push_str("'static ");
                } else if let Some(lt) = &self.ref_lifetime {
                    s.push('\'');
                    s.push_str(lt);
                    s.push(' ');
                }
            }
            RefKind::Mut => {
                s.push('&');
                if use_static {
                    s.push_str("'static ");
                } else if let Some(lt) = &self.ref_lifetime {
                    s.push('\'');
                    s.push_str(lt);
                    s.push(' ');
                }
                s.push_str("mut ");
            }
        }

        // Type name
        s.push_str(&self.type_name);

        // Generic lifetime args
        if !self.type_args.is_empty() {
            s.push('<');
            for (i, arg) in self.type_args.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                if use_static {
                    s.push_str("'static");
                } else {
                    s.push('\'');
                    s.push_str(arg);
                }
            }
            s.push('>');
        }

        s
    }

    /// Reconstruct with lifetimes erased to `'static` (for extern declarations).
    pub fn to_rust_type_static(&self) -> String {
        let mut s = self.to_rust_type_inner(true);
        if self.owned {
            s = format!("Box<{s}>");
        }
        if self.optional {
            s = format!("Option<{s}>");
        }
        s
    }

    /// Convert an owned type to its borrowed equivalent.
    /// `Box<str>` → `&str`. Non-owned types are returned unchanged.
    pub fn as_borrowed(&self) -> TypeRef {
        let mut tr = self.clone();
        if tr.owned {
            tr.owned = false;
            tr.ref_kind = RefKind::Shared;
        }
        tr
    }
}

// ---------------------------------------------------------------------------
// Exported types (structs with #[ffier::export] methods)
// ---------------------------------------------------------------------------

/// A struct exported via `#[ffier::export]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedType {
    /// Rust struct name — key into `type_registry`.
    pub name: String,
    /// FFI destroy function name (e.g. `"ft_widget_destroy"`).
    pub destroy_ffi_name: String,
    /// Whether this type uses by-value self (builder pattern). When true,
    /// by-value self methods in C take a pointer-to-handle so the bridge
    /// can update the handle in place.
    pub is_builder_type: bool,
    pub methods: Vec<Method>,
    /// `#[cfg(...)]` predicate gating this type (e.g. `"feature = \"x\""`,
    /// `"all(feature = \"a\", feature = \"b\")"`). `None` = unconditional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg: Option<String>,
}

// ---------------------------------------------------------------------------
// Methods
// ---------------------------------------------------------------------------

/// A method. Used in exported types, exported traits, and trait impls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Method {
    /// Rust method name (e.g. "get_count").
    pub name: String,
    /// Doc comment lines, verbatim. Each entry is one `///` line.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub doc: Vec<String>,
    pub receiver: Receiver,
    /// Method-level lifetime params with original names (e.g. `["a"]` for `fn foo<'a>(...)`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub method_lifetimes: Vec<String>,
    pub params: Vec<Param>,
    pub ret: Return,
    /// C FFI function name (e.g. `"ft_widget_get_count"`, `"ft_fruit_eat"`).
    pub ffi_name: String,
    /// Present only for trait definition methods (from `#[ffier::export]` on a trait).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trait_definition: Option<TraitMethodDefinition>,
    /// `#[cfg(...)]` predicate gating this method (e.g. `"feature = \"x\""`).
    /// `None` = unconditional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg: Option<String>,
}

/// Extra fields that only exist on trait *definition* methods
/// (from `#[ffier::export]` on a trait), not on concrete impls.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraitMethodDefinition {
    /// Vtable slot index.
    pub index: usize,
    /// Whether this method has a default impl in the trait.
    pub has_default: bool,
}

/// How the method receives `self`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Receiver {
    None,
    Ref,
    Mut,
    Value,
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// A method parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Param {
    /// Parameter name as written in the source.
    pub name: String,
    /// The type and how it's accessed.
    #[serde(flatten)]
    pub param_type: ParamType,
}

/// The type of a parameter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "param_kind", rename_all = "snake_case")]
pub enum ParamType {
    /// Normal parameter — one Rust param maps to one C param.
    Regular(TypeRef),
    /// Slice parameter — one Rust `&[T]` param expanded into N C params
    /// (typically pointer + length). The `c_params` field describes exactly
    /// what the bridge generated at the C ABI level.
    Slice {
        /// Element type reference (e.g. `{ type: "str", ref_kind: "shared" }` for `&[&str]`,
        /// or `{ type: "u8" }` for `&[u8]`).
        element: TypeRef,
        /// The C parameters this single Rust param expanded into.
        /// Consumers use these directly without inferring expansion rules.
        c_params: Vec<CParam>,
    },
    /// `impl Trait` parameter.
    ImplTrait {
        /// Trait name — key into `type_registry`.
        trait_name: String,
        /// Lifetime arguments on the trait at this usage site
        /// (e.g. `["a"]` for `impl Snapshot<'a>`).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        type_args: Vec<String>,
    },
}

/// A C-level parameter produced by slice expansion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CParam {
    /// C parameter name (e.g. `"tags"`, `"tags_len"`).
    pub name: String,
    /// C type (e.g. `"const FtStr*"`, `"uintptr_t"`).
    pub c_type: String,
}

// ---------------------------------------------------------------------------
// Return types
// ---------------------------------------------------------------------------

/// Return type of a method.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "return_kind", rename_all = "snake_case")]
pub enum Return {
    /// Returns nothing.
    Void,
    /// Returns a single value.
    Value(TypeRef),
    /// Returns `Result<T, E>`.
    Result {
        /// The Ok type. `None` when `Result<(), E>`.
        ok: Option<TypeRef>,
        /// Error type name — key into `type_registry`.
        err_type: String,
        /// The C-level calling convention for this Result return.
        /// Determined by the bridge at schema-emission time.
        c_convention: CResultConvention,
    },
    /// Returns `&[T]` or `&[&T]` where T is a handle type.
    /// Encoded as `FfierObjectArray` at the C level.
    ObjectArray {
        /// The element type (the `T` in `&[T]` or `&[&T]`).
        element: TypeRef,
    },
}

/// C-level calling convention for a `Result<T, E>` return.
///
/// Determined once by the bridge and stored in the schema.
/// All generators read this — no independent convention derivation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CResultConvention {
    /// Return `FfierResult` (packed type_tag + error code), ok value via
    /// out-param (`*mut T`), error handle via out-param (`*mut *mut c_void`).
    /// Used for `Result<PrimitiveType, E>`, `Result<(), E>`.
    OutParam,
    /// Return the ok handle pointer directly (`*mut c_void`), `NULL` on error.
    /// Error handle via out-param (`*mut *mut c_void`).
    /// Used for `Result<HandleType, E>`.
    HandleOrNull,
}

/// Well-known type name for builder methods that return `Self`.
pub const SELF_TYPE: &str = "Self";

/// Well-known type name for `*mut core::ffi::c_void` (opaque mutable pointer).
pub const C_VOID_PTR: &str = "*mut core::ffi::c_void";

/// Well-known type name for `*const core::ffi::c_void` (opaque const pointer).
pub const C_VOID_CONST_PTR: &str = "*const core::ffi::c_void";

/// Check whether a type name refers to a `ReplacesSelf` sentinel in the registry.
fn is_replaces_self(type_name: &str, registry: &BTreeMap<String, TypeEntry>) -> bool {
    registry
        .get(type_name)
        .is_some_and(|e| e.bless == Some(Blessing::ReplacesSelf))
}

impl Return {
    /// True if this return type represents a builder pattern (`-> Self` or
    /// `-> Result<Self, E>`), detected via `Blessing::ReplacesSelf` in the
    /// type registry.
    pub fn is_builder_self(&self, registry: &BTreeMap<String, TypeEntry>) -> bool {
        match self {
            Return::Value(tr) => is_replaces_self(&tr.type_name, registry),
            Return::Result { ok: Some(tr), .. } => is_replaces_self(&tr.type_name, registry),
            _ => false,
        }
    }

    /// Reconstruct the Rust return type syntax.
    /// Types with `Blessing::ReplacesSelf` in the registry render as `Self`.
    pub fn to_rust_type(&self, registry: &BTreeMap<String, TypeEntry>) -> String {
        match self {
            Return::Void => "()".to_string(),
            Return::Value(tr) if is_replaces_self(&tr.type_name, registry) => "Self".to_string(),
            Return::Value(tr) => tr.to_rust_type(),
            Return::Result { ok, err_type, .. } => {
                let ok_str = match ok {
                    Some(tr) if is_replaces_self(&tr.type_name, registry) => "Self".to_string(),
                    Some(tr) => tr.to_rust_type(),
                    None => "()".to_string(),
                };
                format!("Result<{ok_str}, {err_type}>")
            }
            Return::ObjectArray { element } => {
                format!("ForeignSlice<{}>", element.to_rust_type())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// An error enum exported via `#[derive(FfiError)]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorType {
    /// Rust enum name — key into `type_registry`.
    pub name: String,
    pub variants: Vec<ErrorVariant>,
}

/// A variant of an error enum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorVariant {
    /// Variant name (e.g. "NotFound").
    pub name: String,
    /// C constant name (e.g. "FT_ERROR_TEST_NOT_FOUND").
    pub c_name: String,
    /// Numeric error code from `#[ffier(code = N)]`.
    pub code: u32,
    /// Human-readable message.
    pub message: String,
    /// Payload fields carried by this variant. Empty for unit variants.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fields: Vec<ErrorField>,
}

/// A payload field in a data-carrying error variant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorField {
    /// Field type reference (e.g. `str` for `Box<str>`).
    #[serde(flatten)]
    pub type_ref: TypeRef,
}

// ---------------------------------------------------------------------------
// Enum constants
// ---------------------------------------------------------------------------

/// A plain enum exported as C `#define` constants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumType {
    /// Rust enum name — key into `type_registry`.
    pub name: String,
    pub variants: Vec<EnumVariant>,
    /// `#[cfg(...)]` predicate gating this enum. `None` = unconditional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg: Option<String>,
}

/// A variant of an enum constant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnumVariant {
    /// Variant name (e.g. "Off").
    pub name: String,
    /// C constant name (e.g. "FT_LOG_LEVEL_OFF").
    pub c_name: String,
    /// Numeric discriminant value.
    pub value: u64,
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// A free function exported via `#[ffier::export]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeFunction {
    /// Rust function name (e.g. "init_log").
    pub name: String,
    /// C FFI function name (e.g. "ft_init_log").
    pub ffi_name: String,
    /// Doc comment lines.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub doc: Vec<String>,
    pub params: Vec<Param>,
    pub ret: Return,
    /// `#[cfg(...)]` predicate gating this function. `None` = unconditional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg: Option<String>,
}

// ---------------------------------------------------------------------------
// Exported traits
// ---------------------------------------------------------------------------

/// A trait exported via `#[ffier::export]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementableTrait {
    /// Trait name — key into `type_registry`.
    pub name: String,
    /// FFI destroy/dispatch function name (e.g. `"ft_fruit_destroy"`).
    pub destroy_ffi_name: String,
    /// C constant name for the vtable handle type tag
    /// (e.g. `"FT_PUSH_STR_TYPE_TAG"`).
    pub type_tag_constant: String,
    /// C vtable struct name (e.g. `"FtFruitVtable"`).
    pub vtable_struct_c_name: String,
    /// C wrapper type name (e.g. `"FtVtableFruit"`).
    pub wrapper_c_name: String,
    /// Rust vtable struct name (e.g. `"FruitVtable"`).
    pub vtable_struct_name: String,
    /// Rust wrapper type name (e.g. `"VtableFruit"`).
    pub wrapper_name: String,

    pub methods: Vec<Method>,
    /// Number of methods that belong to this trait (not supertrait methods).
    pub own_method_count: usize,
    /// Highest vtable slot index (including reserved/retired slots).
    pub max_vtable_slot: usize,
    /// If true, no vtable wrapper type exists. C callers cannot implement
    /// this trait — only concrete Rust implementors are dispatched.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub no_vtable: bool,
    /// `#[cfg(...)]` predicate gating this trait. `None` = unconditional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg: Option<String>,
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

/// An `impl Trait for Struct` exported via `#[ffier::export]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitImpl {
    /// Trait name — key into `type_registry`.
    pub trait_name: String,
    /// Struct name — key into `type_registry`.
    pub struct_name: String,
    /// Lifetime params on the impl block.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lifetimes: Vec<String>,
    /// Lifetime args on the trait.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trait_lifetime_args: Vec<String>,
    /// Lifetime args on the struct.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub struct_lifetime_args: Vec<String>,
    pub methods: Vec<Method>,
    /// `#[cfg(...)]` predicate gating this trait impl. `None` = unconditional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg: Option<String>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_ref_none(v: &RefKind) -> bool {
    *v == RefKind::None
}

impl Library {
    /// The effective prefix for primitive types (Str, Bytes, Result, etc.).
    /// Falls back to the library prefix if not explicitly set.
    pub fn primitives_prefix(&self) -> &str {
        self.primitives_prefix.as_deref().unwrap_or(&self.prefix)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("failed to serialize library metadata")
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Look up a type entry by name.
    pub fn type_entry(&self, name: &str) -> Option<&TypeEntry> {
        self.type_registry.get(name)
    }

    /// Get the C type name for a type by its registry key, resolving
    /// through alias/enum chains to the concrete C type.
    /// Panics if the type is not in the registry or if the alias chain
    /// exceeds 16 hops (indicates a cycle).
    pub fn c_type_of(&self, name: &str) -> &str {
        const MAX_DEPTH: usize = 16;
        let mut current = name;
        for _ in 0..MAX_DEPTH {
            let entry = self
                .type_registry
                .get(current)
                .unwrap_or_else(|| panic!("type `{current}` not found in type_registry"));
            match &entry.kind {
                TypeKind::Alias { alias_of }
                | TypeKind::Enum { alias_of }
                | TypeKind::Bitflags { alias_of } => {
                    current = alias_of;
                }
                TypeKind::Primitive { c_type } => return c_type,
                TypeKind::String { c_name }
                | TypeKind::Bytes { c_name }
                | TypeKind::Handle { c_name }
                | TypeKind::Error { c_name }
                | TypeKind::Trait { c_name } => return c_name,
                TypeKind::ForeignHandle { c_name, .. } => return c_name,
            }
        }
        panic!("alias chain for `{name}` exceeds {MAX_DEPTH} hops — probable cycle");
    }

    /// Collect all type names referenced by methods, params, returns, errors,
    /// enums, and aliases in this library. Used to prune unreferenced types.
    pub fn referenced_types(&self) -> std::collections::HashSet<&str> {
        let mut refs = std::collections::HashSet::new();

        fn collect_from_params<'a>(
            params: &'a [Param],
            refs: &mut std::collections::HashSet<&'a str>,
        ) {
            for p in params {
                match &p.param_type {
                    ParamType::Regular(tr) => {
                        refs.insert(&tr.type_name);
                    }
                    ParamType::Slice { element, .. } => {
                        refs.insert(&element.type_name);
                    }
                    ParamType::ImplTrait { trait_name, .. } => {
                        refs.insert(trait_name);
                    }
                }
            }
        }

        fn collect_from_return<'a>(ret: &'a Return, refs: &mut std::collections::HashSet<&'a str>) {
            match ret {
                Return::Void => {}
                Return::Value(tr) => {
                    refs.insert(&tr.type_name);
                }
                Return::Result { ok, err_type, .. } => {
                    if let Some(tr) = ok {
                        refs.insert(&tr.type_name);
                    }
                    refs.insert(err_type);
                }
                Return::ObjectArray { element } => {
                    refs.insert(&element.type_name);
                }
            }
        }

        fn collect_from_methods<'a>(
            methods: &'a [Method],
            refs: &mut std::collections::HashSet<&'a str>,
        ) {
            for m in methods {
                collect_from_params(&m.params, refs);
                collect_from_return(&m.ret, refs);
            }
        }

        for ty in &self.exported_types {
            refs.insert(ty.name.as_str());
            collect_from_methods(&ty.methods, &mut refs);
        }
        for err in &self.errors {
            refs.insert(err.name.as_str());
        }
        for en in &self.enum_constants {
            refs.insert(en.name.as_str());
        }
        for bf in &self.bitflags_constants {
            refs.insert(bf.name.as_str());
        }
        for f in &self.free_functions {
            collect_from_params(&f.params, &mut refs);
            collect_from_return(&f.ret, &mut refs);
        }
        for tr in &self.traits {
            refs.insert(tr.name.as_str());
            collect_from_methods(&tr.methods, &mut refs);
        }
        for ti in &self.trait_impls {
            refs.insert(ti.trait_name.as_str());
            refs.insert(ti.struct_name.as_str());
            collect_from_methods(&ti.methods, &mut refs);
        }

        // Collect all methods for implicit framework type detection.
        let all_methods = || {
            self.exported_types
                .iter()
                .flat_map(|t| t.methods.iter())
                .chain(self.traits.iter().flat_map(|t| t.methods.iter()))
                .chain(self.trait_impls.iter().flat_map(|t| t.methods.iter()))
        };

        let has_result = all_methods().any(|m| matches!(m.ret, Return::Result { .. }))
            || self
                .free_functions
                .iter()
                .any(|f| matches!(f.ret, Return::Result { .. }));

        let has_impl_trait = all_methods().any(|m| {
            m.params
                .iter()
                .any(|p| matches!(p.param_type, ParamType::ImplTrait { .. }))
        }) || self.free_functions.iter().any(|f| {
            f.params
                .iter()
                .any(|p| matches!(p.param_type, ParamType::ImplTrait { .. }))
        });

        // If any Result-returning method exists, the error trait and result
        // framework type are implicitly referenced.
        if has_result {
            for (name, entry) in &self.type_registry {
                if matches!(
                    entry.bless,
                    Some(Blessing::Result) | Some(Blessing::ErrorTrait)
                ) {
                    refs.insert(name.as_str());
                }
            }
        }

        // If any impl Trait param exists, the vtable handle is implicitly referenced.
        if has_impl_trait {
            for (name, entry) in &self.type_registry {
                if entry.bless == Some(Blessing::VtableHandle) {
                    refs.insert(name.as_str());
                }
            }
        }

        // Walk alias chains — if X is referenced and X aliases Y, Y is also referenced.
        let mut changed = true;
        while changed {
            changed = false;
            let snapshot: Vec<&str> = refs.iter().copied().collect();
            for name in snapshot {
                if let Some(entry) = self.type_registry.get(name)
                    && let TypeKind::Alias { alias_of }
                    | TypeKind::Enum { alias_of }
                    | TypeKind::Bitflags { alias_of } = &entry.kind
                    && refs.insert(alias_of)
                {
                    changed = true;
                }
            }
        }

        refs
    }

    /// Remove unreferenced types from the registry.
    pub fn prune_unreferenced_types(&mut self) {
        let refs: std::collections::HashSet<String> = self
            .referenced_types()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        self.type_registry.retain(|name, entry| {
            refs.contains(name.as_str())
                || entry.bless == Some(Blessing::Object)
                || entry.bless == Some(Blessing::ObjectArray)
        });
    }

    /// Find the unique type with the given blessing.
    /// Panics if more than one type carries the same blessing.
    pub fn blessed(&self, tag: Blessing) -> Option<(&str, &TypeEntry)> {
        let mut iter = self
            .type_registry
            .iter()
            .filter(|(_, entry)| entry.bless == Some(tag));
        let result = iter.next();
        if let Some((dup, _)) = iter.next() {
            panic!("multiple types blessed as {tag:?} (duplicate: `{dup}`)");
        }
        result.map(|(name, entry)| (name.as_str(), entry))
    }
}
